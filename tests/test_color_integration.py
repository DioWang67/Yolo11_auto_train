
import sys
import unittest
from unittest.mock import MagicMock, patch

# Shield C++ fatal DLLs explicitly for GUI mock resolution
class MockTensor:
    pass
torch_mock = MagicMock()
torch_mock.Tensor = MockTensor
sys.modules["torch"] = torch_mock
sys.modules["ultralytics"] = MagicMock()
sys.modules["segment_anything"] = MagicMock()
sys.modules["albumentations"] = MagicMock()
# Mock sklearn structure
# sklearn = MagicMock()
# sys.modules["sklearn"] = sklearn
# sys.modules["sklearn.model_selection"] = MagicMock()

# ``color_panel`` imports the color verifier at module scope, so the stand-in
# has to be in place before that import -- this suite exercises the GUI panel,
# not the gate. It is put back immediately afterwards: the substitution used to
# stay in ``sys.modules`` for the rest of the session, so whichever suite ran
# after this one tested a MagicMock instead of the real color gate, and did it
# silently, as passes. A conformance guard that can be switched off by test
# ordering is not a guard.
_VERIFIER_MODULE = "picture_tool.color.color_verifier"
_REAL_VERIFIER_MODULE = sys.modules.get(_VERIFIER_MODULE)
sys.modules[_VERIFIER_MODULE] = MagicMock()

from PyQt5.QtWidgets import QApplication  # noqa: E402
from picture_tool.gui.color_panel import ColorPanel  # noqa: E402

# Importing the panel bound the stand-in in two places: ``sys.modules`` and, as
# the import machinery always does, an attribute of the parent package -- which
# ``picture_tool.color``'s lazy ``__getattr__`` also caches in its globals. A
# later ``from picture_tool.color import color_verifier`` reads that attribute,
# so restoring only ``sys.modules`` leaves the mock in place.
_VERIFIER_PACKAGE, _, _VERIFIER_ATTR = _VERIFIER_MODULE.rpartition(".")
if _REAL_VERIFIER_MODULE is not None:
    sys.modules[_VERIFIER_MODULE] = _REAL_VERIFIER_MODULE
else:
    sys.modules.pop(_VERIFIER_MODULE, None)
_verifier_package = sys.modules.get(_VERIFIER_PACKAGE)
if _verifier_package is not None:
    if _REAL_VERIFIER_MODULE is not None:
        setattr(_verifier_package, _VERIFIER_ATTR, _REAL_VERIFIER_MODULE)
    else:
        # Dropping the cache makes the lazy loader import the real module on the
        # next request, instead of handing back the stand-in for good.
        try:
            delattr(_verifier_package, _VERIFIER_ATTR)
        except AttributeError:
            pass

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

    @patch("picture_tool.color.color_inspection.run_gui_session")
    @patch("PyQt5.QtWidgets.QDialog.exec_", return_value=1) # 1 is QDialog.Accepted
    @patch("PyQt5.QtWidgets.QMessageBox.warning")
    @patch("PyQt5.QtWidgets.QMessageBox.critical")
    @patch("PyQt5.QtCore.QSettings")
    def test_sam_launch_memory(self, mock_settings_cls, mock_crit, mock_warn, mock_exec, mock_run_session):
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
        with patch("pathlib.Path.exists", return_value=True):
            
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
            mock_run_session.assert_called_once()
            call_args = mock_run_session.call_args[0][0]
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
            
            # Wait for thread completion to avoid mock timing assertion failure
            if hasattr(panel, "verify_worker"):
                panel.verify_worker.wait()
            QApplication.processEvents()
            
            # Check call
            mock_verifier.verify_directory.assert_called_once()
            
            # Check output in result_text
            text = panel.result_text.toPlainText()
            self.assertIn("驗證完成", text)

if __name__ == "__main__":
    unittest.main()
