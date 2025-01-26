import unittest
from unittest.mock import Mock, patch

import requests

from llm_interface.remote_ollama import RemoteOllama


class TestRemoteOllama(unittest.TestCase):
    def setUp(self):
        self.mock_ssh = Mock()
        self.remote_ollama = RemoteOllama(self.mock_ssh, "test-model")

    def test_install_ollama(self):
        self.mock_ssh.execute_command.return_value = {"exit_status": 0, "stderr": ""}
        self.remote_ollama.install_ollama()
        expected_calls = [
            unittest.mock.call("sudo apt-get update"),
            unittest.mock.call("sudo apt-get install -y curl"),
            unittest.mock.call("curl https://ollama.ai/install.sh | sh"),
        ]
        self.mock_ssh.execute_command.assert_has_calls(expected_calls)
        self.assertEqual(self.mock_ssh.execute_command.call_count, 3)

    def test_install_ollama_failure(self):
        self.mock_ssh.execute_command.return_value = {
            "exit_status": 1,
            "stderr": "Error",
        }
        with self.assertRaises(Exception):
            self.remote_ollama.install_ollama()

    def test_pull_model(self):
        self.mock_ssh.execute_command.return_value = {"exit_status": 0, "stderr": ""}
        self.remote_ollama.pull_model()
        self.mock_ssh.execute_command.assert_called_once_with(
            "/usr/local/bin/ollama pull test-model"
        )

    def test_pull_model_failure(self):
        self.mock_ssh.execute_command.return_value = {
            "exit_status": 1,
            "stderr": "Error",
        }
        with self.assertRaises(Exception):
            self.remote_ollama.pull_model()

    def test_start_server(self):
        self.mock_ssh.execute_command.return_value = {"exit_status": 0, "stderr": ""}
        self.remote_ollama.start_server()
        self.mock_ssh.execute_command.assert_called_once_with(
            "nohup /usr/local/bin/ollama serve > /dev/null 2>&1 &"
        )
        self.mock_ssh.start_port_forward.assert_called_once_with(
            11434, "localhost", 11434
        )

    def test_start_server_failure(self):
        self.mock_ssh.execute_command.return_value = {
            "exit_status": 1,
            "stderr": "Error",
        }
        with self.assertRaises(Exception):
            self.remote_ollama.start_server()

    @patch("time.sleep")
    @patch("requests.get")
    def test_validate_server_success(self, mock_get, mock_sleep):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = self.remote_ollama.validate_server()

        self.assertTrue(result)
        mock_get.assert_called_once_with("http://localhost:11434/api/tags", timeout=5)
        mock_sleep.assert_not_called()

    @patch("time.sleep")
    @patch("requests.get")
    def test_validate_server_failure(self, mock_get, mock_sleep):
        mock_get.side_effect = requests.RequestException

        result = self.remote_ollama.validate_server()

        self.assertFalse(result)
        self.assertEqual(mock_get.call_count, 5)
        self.assertEqual(mock_sleep.call_count, 5)

    def test_stop_server(self):
        self.mock_ssh.execute_command.return_value = {"exit_status": 0, "stderr": "OK"}
        self.remote_ollama.stop_server()
        self.mock_ssh.execute_command.assert_called_once_with("pkill ollama")
        self.mock_ssh.stop_port_forward.assert_called_once()


if __name__ == "__main__":
    unittest.main()
