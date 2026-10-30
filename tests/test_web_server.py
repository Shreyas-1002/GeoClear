"""Unit test suite for GeoClear FastAPI web server and REST APIs."""

import unittest
from fastapi.testclient import TestClient

from server import app, state


class TestGeoClearWebServer(unittest.TestCase):
    """Test suite verifying FastAPI endpoints, data ingestion, and inference workflows."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_index_html_served(self):
        """GET / must serve the single-page HTML interface containing GeoClear branding."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("GeoClear", res.text)
        self.assertIn("Multimodal Satellite Cloud Removal", res.text)

    def test_02_status_endpoint(self):
        """GET /api/status must return operational status."""
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ready")
        self.assertIn("selected_model_id", data)

    def test_03_models_endpoint_returns_two_operational_models(self):
        """GET /api/models must return only ECRformer-Light and DSen2-CR."""
        res = self.client.get("/api/models")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        model_ids = [m["model_id"] for m in data["models"]]
        self.assertEqual(set(model_ids), {"ecrformer", "dsen2_cr"})
        self.assertNotIn("diffcr", model_ids)
        self.assertNotIn("cloudgan", model_ids)

    def test_04_model_selection(self):
        """POST /api/models/select must switch the active model."""
        res = self.client.post("/api/models/select", data={"model_id": "dsen2_cr"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(state.selected_model_id, "dsen2_cr")

        # Switch back to ecrformer
        res = self.client.post("/api/models/select", data={"model_id": "ecrformer"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(state.selected_model_id, "ecrformer")

    def test_05_data_declaration(self):
        """POST /api/declare must evaluate input compatibility against operational models."""
        res = self.client.post(
            "/api/declare",
            data={
                "has_s1": True,
                "has_s2": True,
                "optical_bands": "13",
                "sar_vv": True,
                "sar_vh": True,
                "temporal_count": 1,
                "has_reference": True,
                "height": 256,
                "width": 256,
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["compatible_count"], 2)
        compat_ids = [m["model_id"] for m in data["compatible"]]
        self.assertIn("ecrformer", compat_ids)
        self.assertIn("dsen2_cr", compat_ids)

    def test_06_inspect_local_patch(self):
        """POST /api/ingest/inspect_local must validate data/real_patch non-destructively."""
        res = self.client.post("/api/ingest/inspect_local", data={"folder_path": "data/real_patch"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["is_valid"])
        self.assertEqual(data["sar_file"], "s1_sar.tif")
        self.assertEqual(data["opt_file"], "s2_cloudy.tif")
        self.assertIsNotNone(data["manifest"])
        self.assertIn("Mumbai", data["manifest"]["dataset_name"])

    def test_07_load_local_patch(self):
        """POST /api/ingest/load_local must load arrays into server memory and generate previews."""
        res = self.client.post("/api/ingest/load_local", data={"folder_path": "data/real_patch"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["sar_shape"], [2, 256, 256])
        self.assertEqual(data["opt_shape"], [13, 256, 256])
        self.assertTrue(data["cloudy_preview"].startswith("data:image/png;base64,"))
        self.assertTrue(data["sar_preview"].startswith("data:image/png;base64,"))

    def test_08_reconstruction_and_evaluation(self):
        """POST /api/reconstruct must execute forward pass on CPU and return previews & metrics."""
        # Ensure patch is loaded
        self.client.post("/api/ingest/load_local", data={"folder_path": "data/real_patch"})

        # Run ECRformer inference
        res = self.client.post("/api/reconstruct", data={"model_id": "ecrformer"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["model_id"], "ecrformer")
        self.assertTrue(data["finite_valid"])
        self.assertGreater(data["latency_ms"], 0)
        self.assertTrue(data["reconstructed_preview"].startswith("data:image/png;base64,"))

        # Check metrics if reference was provided
        if data["metrics"] is not None:
            self.assertIn("mae", data["metrics"])
            self.assertIn("rgb_psnr", data["metrics"])
            self.assertIn("rgb_ssim", data["metrics"])
            self.assertEqual(len(data["metrics"]["per_band_mae"]), 13)

    def test_09_multispectral_band_explorer(self):
        """GET /api/bands must return all 13 spectral band previews and metadata."""
        res = self.client.get("/api/bands")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data["bands"]), 13)
        self.assertEqual(data["bands"][0]["band_code"], "B1")
        self.assertEqual(data["bands"][10]["band_code"], "B10")
        self.assertTrue(data["bands"][0]["image"].startswith("data:image/png;base64,"))

    def test_10_historical_benchmarks(self):
        """GET /api/benchmarks/munich and /api/benchmarks/mumbai must return verified results."""
        res_munich = self.client.get("/api/benchmarks/munich")
        self.assertEqual(res_munich.status_code, 200)
        data_m = res_munich.json()
        self.assertIn("Munich", data_m["city"])
        self.assertEqual(data_m["models"]["ecrformer"]["rgb_psnr"], 32.95)

        res_mumbai = self.client.get("/api/benchmarks/mumbai")
        self.assertEqual(res_mumbai.status_code, 200)
        data_mum = res_mumbai.json()
        self.assertIn("Mumbai", data_mum["city"])
        self.assertEqual(data_mum["models"]["ecrformer"]["rgb_psnr"], 29.62)

    def test_11_export_endpoints(self):
        """GET /api/export/* must return valid file downloads."""
        res_npz = self.client.get("/api/export/npz")
        self.assertEqual(res_npz.status_code, 200)
        self.assertGreater(len(res_npz.content), 100)

        res_png = self.client.get("/api/export/png")
        self.assertEqual(res_png.status_code, 200)
        self.assertGreater(len(res_png.content), 100)


if __name__ == "__main__":
    unittest.main()
