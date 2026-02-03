
import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Mock dependencies to allow import in CI/Headless environments
sys.modules["cv2"] = MagicMock()
sys.modules["segment_anything"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["albumentations"] = MagicMock()
sys.modules["ultralytics"] = MagicMock()
sys.modules["pandas"] = MagicMock()
sys.modules["ultralytics"] = MagicMock()
sys.modules["pandas"] = MagicMock()
sys.modules["yaml"] = MagicMock()

# Mock matplotlib structure
mpl = MagicMock()
sys.modules["matplotlib"] = mpl
sys.modules["matplotlib.pyplot"] = MagicMock()
sys.modules["matplotlib.patches"] = MagicMock()

sys.modules["scipy"] = MagicMock()

# Mock sklearn structure
sklearn = MagicMock()
sys.modules["sklearn"] = sklearn
sys.modules["sklearn.model_selection"] = MagicMock()

# Mock internal heavy modules
sys.modules["picture_tool.color.color_inspection"] = MagicMock()
# We want to test color_verifier logic, so we might not want to mock it entirely, 
# but if it depends on cv2, we have to rely on mocks or careful imports.
# For integration test of the GUI Panel, we can mock it.
sys.modules["picture_tool.color.color_verifier"] = MagicMock()

from PyQt5.QtWidgets import QApplication
from picture_tool.gui.color_panel import ColorPanel

class TestColorIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not QApplication.instance():
            cls.app = QApplication([])
        else:
            cls.app = QApplication.instance()

    def test_panel_instantiation(self):
        """Test that ColorPanel can be instantiated."""
        manager = MagicMock()
        panel = ColorPanel(manager)
        self.assertIsNotNone(panel)
        # Check tabs
        self.assertEqual(panel.tabs.count(), 2)
        self.assertEqual(panel.tabs.tabText(0), "🎨 顏色範本蒐集 (SAM)")
        self.assertEqual(panel.tabs.tabText(1), "✅ 批次顏色驗證")

    def test_verification_tab_ui(self):
        """Test existence of UI elements in Verification tab."""
        manager = MagicMock()
        panel = ColorPanel(manager)
        
        # Verify UI elements exist (accessed via internal names or layout)
        self.assertTrue(hasattr(panel, "verify_input_edit"))
        self.assertTrue(hasattr(panel, "verify_stats_edit"))
        self.assertTrue(hasattr(panel, "result_text"))

    @patch("picture_tool.gui.color_panel.color_inspection")
    @patch("PyQt5.QtWidgets.QDialog.exec_", return_value=1) # 1 is QDialog.Accepted
    @patch("PyQt5.QtWidgets.QMessageBox.warning")
    @patch("PyQt5.QtWidgets.QMessageBox.critical")
    @patch("PyQt5.QtCore.QSettings")
    def test_sam_launch_memory(self, mock_settings_cls, mock_crit, mock_warn, mock_exec, mock_insp):
        """Test that SAM launcher loads and saves settings."""
        manager = MagicMock()
        panel = ColorPanel(manager)
        
        # Mock QSettings instance
        mock_settings = MagicMock()
        mock_settings_cls.return_value = mock_settings
        
        mock_settings.value.side_effect = lambda key, default=None, **kwargs: {
            "sam_tool/input_dir": "/tmp/in",
            "sam_tool/output_json": "/tmp/out.json",
            "sam_tool/checkpoint": "/tmp/model.pth",
            "sam_tool/model_type": "vit_l",
            "sam_tool/target_colors": "Gold, Silver",
            "use_cuda": False
        }.get(key, default)
        
        # Mock Path.exists for checkpoint check
        with patch("pathlib.Path.exists", return_value=True), \
             patch("torch.cuda.is_available", return_value=False):
            
            # This triggers the dialog logic
            # We need to access the dialog inputs to 'simulate' user typing if values weren't loaded
            # But since we mock load, the inputs should be pre-filled.
            # Then exec_ returns Accepted, so it should trigger save.
            
            panel._launch_sam_tool()
            
            # Verify values were loaded into widgets (indirectly via what is saved back)
            # The logic reads from widgets to save. So if load worked, widgets have correct text.
            # And then save should write that text back.
            
            # Verify save calls
            mock_settings.setValue.assert_any_call("sam_tool/input_dir", "/tmp/in")
            mock_settings.setValue.assert_any_call("sam_tool/output_json", "/tmp/out.json")
            mock_settings.setValue.assert_any_call("sam_tool/checkpoint", "/tmp/model.pth")
            mock_settings.setValue.assert_any_call("sam_tool/model_type", "vit_l")
            mock_settings.setValue.assert_any_call("sam_tool/target_colors", "Gold, Silver")
            
            # Check if we hit early returns
            if mock_warn.called:
                print(f"WARNING CALLED: {mock_warn.call_args}")
            if mock_crit.called:
                print(f"CRITICAL CALLED: {mock_crit.call_args}")
                
            mock_warn.assert_not_called()
            mock_crit.assert_not_called()

            # Verify run_gui_session called with correct config
            mock_insp.run_gui_session.assert_called_once()
            call_args = mock_insp.run_gui_session.call_args[0][0]
            self.assertEqual(call_args.sam.model_type, "vit_l")
            self.assertEqual(call_args.colors, ["Gold", "Silver"])

    @patch("picture_tool.gui.color_panel.color_verifier")
    def test_run_verification_trigger(self, mock_verifier):
        """Test that clicking run triggers the verification logic."""
        manager = MagicMock()
        panel = ColorPanel(manager)
        
        # Setup inputs
        panel.verify_input_edit.setText("dummy_dir")
        panel.verify_stats_edit.setText("dummy.json")
        
        # Mock Path.exists to return True
        with patch("pathlib.Path.exists", return_value=True):
            # Mock return values for verify_directory
            mock_verifier.verify_directory.return_value = ({'total': 10}, [])
            
            # Trigger
            panel._run_verification()
            
            # Check call
            mock_verifier.verify_directory.assert_called_once()
            
            # Check output in result_text
            text = panel.result_text.toPlainText()
            self.assertIn("驗證完成", text)

if __name__ == "__main__":
    unittest.main()
