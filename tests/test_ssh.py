import unittest
from unittest.mock import Mock, patch

from llm_interface.ssh import SSHConnection


class TestSSHConnection(unittest.TestCase):
    def setUp(self):
        self.ssh = SSHConnection("example.com", "user")

    @patch("llm_interface.ssh.paramiko.SSHClient")
    def test_connect(self, mock_ssh_client):
        # Set up agent keys mock
        mock_agent = Mock()
        mock_keys = [Mock()]
        mock_agent.get_keys.return_value = mock_keys

        with patch("paramiko.Agent", return_value=mock_agent):
            self.ssh.connect()
            mock_ssh_client.return_value.connect.assert_called_once_with(
                "example.com", port=22, username="user", pkey=mock_keys[0]
            )

    @patch("llm_interface.ssh.paramiko.SSHClient")
    def test_execute_command(self, mock_ssh_client):
        mock_channel = Mock()
        mock_channel.recv_exit_status.return_value = 0
        mock_stdout = Mock()
        mock_stdout.channel = mock_channel
        mock_stdout.read.return_value = b"Command output"
        mock_stderr = Mock()
        mock_stderr.read.return_value = b""
        mock_ssh_client.return_value.exec_command.return_value = (
            None,
            mock_stdout,
            mock_stderr,
        )

        self.ssh.client = mock_ssh_client.return_value
        result = self.ssh.execute_command("test command")

        self.assertEqual(result["exit_status"], 0)
        self.assertEqual(result["stdout"], "Command output")
        self.assertEqual(result["stderr"], "")

    @patch("llm_interface.ssh.paramiko.SSHClient")
    def test_stop_port_forward(self, mock_ssh_client):
        mock_transport = Mock()
        mock_ssh_client.return_value.get_transport.return_value = mock_transport
        self.ssh.client = mock_ssh_client.return_value
        mock_thread = Mock()
        self.ssh.forward_thread = mock_thread

        self.ssh.stop_port_forward()

        mock_transport.cancel_port_forward.assert_called_once_with("", 0)
        mock_thread.join.assert_called_once()
        self.assertIsNone(self.ssh.forward_thread)


if __name__ == "__main__":
    unittest.main()
