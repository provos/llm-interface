import json
import unittest
from typing import Optional
from unittest.mock import Mock, patch

from ollama import Client
from pydantic import BaseModel, Field

import llm_interface.errors as errors
from llm_interface import LLMInterface, ModelError, TokenUsage, tool
from llm_interface.testing.helpers import MockCache


@tool(name="mock_tool")
def mock_function(param1: str, param2: int) -> str:
    return f"Executed mock function with args: {param1}, {param2}"


class DummyPydanticModel(BaseModel):
    field1: str
    field2: int


class TestLLMInterface(unittest.TestCase):
    def setUp(self):
        # Mock the Client
        self.mock_client = Mock(spec=Client)

        # Initialize the LLMInterface with InMemoryCache
        self.llm_interface = LLMInterface(client=self.mock_client)
        self.llm_interface.disk_cache = MockCache()

        self.prompt = "What is the capital of France?"
        self.system = "test_system"
        self.response_content = "Paris"
        self.response_data = {"message": {"content": self.response_content}}

    def test_generate_pydantic_valid_response(self):
        output_model = DummyPydanticModel(field1="test", field2=42)
        valid_json_response = '{"field1": "test", "field2": 42}'
        self.mock_client.chat.return_value = {
            "message": {"content": valid_json_response}
        }

        response = self.llm_interface.generate_pydantic(
            prompt_template=self.prompt,
            output_schema=DummyPydanticModel,
            system=self.system,
        )

        self.assertEqual(response, output_model)

    def test_generate_pydantic_invalid_response(self):
        # Simulate an invalid JSON that cannot be parsed into DummyPydanticModel
        invalid_json_response = '{"field1": "test"}'
        self.mock_client.chat.return_value = {
            "message": {"content": invalid_json_response}
        }

        response = self.llm_interface.generate_pydantic(
            prompt_template=self.prompt,
            output_schema=DummyPydanticModel,
            system=self.system,
        )

        self.assertIsNone(response)  # Expecting None due to parsing error

    def test_generate_pydantic_with_retry_logic_and_prompt_check(self):
        # Simulate an invalid JSON response that fails to parse initially
        invalid_content = '{"field1": "test"}'
        valid_content = '{"field1": "valid", "field2": 42}'

        self.mock_client.chat.side_effect = [
            {"message": {"content": invalid_content}},
            {"message": {"content": valid_content}},
        ]

        output_model = DummyPydanticModel(field1="valid", field2=42)

        with patch("llm_interface.llm_interface.logging.Logger") as mock_logger:
            self.llm_interface.logger = mock_logger

            response = self.llm_interface.generate_pydantic(
                prompt_template=self.prompt,
                output_schema=DummyPydanticModel,
                system=self.system,
            )

            # Assert the method eventually returns the correct output after retry
            self.assertEqual(response, output_model)
            self.assertEqual(
                self.mock_client.chat.call_count, 2
            )  # Should be called twice due to retry

            # Check the messages passed to the chat function of the client
            first_call_messages = self.mock_client.chat.call_args_list[0][1]["messages"]
            second_call_messages = self.mock_client.chat.call_args_list[1][1][
                "messages"
            ]

            # Assert the second message includes instruction for correcting the format
            self.assertIn("field2", second_call_messages[3]["content"])
            self.assertNotIn(
                "field2", first_call_messages[1]["content"]
            )  # Ensure initial message was clean

    def test_get_format_instructions(self):
        instructions = LLMInterface.get_format_instructions(DummyPydanticModel)

        # Check if the instructions contain the expected elements
        self.assertIn("The output should be formatted as a JSON instance", instructions)
        self.assertIn("Here is the output schema:", instructions)

        # Check if the schema contains the fields from DummyPydanticModel
        self.assertIn('"field1":', instructions)
        self.assertIn('"type": "string"', instructions)
        self.assertIn('"field2":', instructions)
        self.assertIn('"type": "integer"', instructions)

        # Verify that the schema is valid JSON
        try:
            schema_start = instructions.index("```\n") + 4
            schema_end = instructions.rindex("\n```")
            json_schema = instructions[schema_start:schema_end]
            parsed_schema = json.loads(json_schema)

            # Check if parsed schema has the expected structure
            self.assertIn("properties", parsed_schema)
            self.assertIn("field1", parsed_schema["properties"])
            self.assertIn("field2", parsed_schema["properties"])
        except json.JSONDecodeError:
            self.fail("The schema in the instructions is not valid JSON")

    def test_generate_pydantic_with_debug_saver(self):
        output_model = DummyPydanticModel(field1="test", field2=42)
        valid_json_response = '{"field1": "test", "field2": 42}'
        self.mock_client.chat.return_value = {
            "message": {"content": valid_json_response}
        }

        # Mock debug_saver function
        mock_debug_saver = Mock()

        # Additional kwargs for the prompt template
        additional_kwargs = {"extra_param": "value"}

        response = self.llm_interface.generate_pydantic(
            prompt_template="Test prompt with {extra_param}",
            output_schema=DummyPydanticModel,
            system=self.system,
            debug_saver=mock_debug_saver,
            **additional_kwargs,
        )

        self.assertEqual(response, output_model)

        # Assert that debug_saver was called with the correct arguments
        mock_debug_saver.assert_called_once()
        call_args = mock_debug_saver.call_args[0]

        # Check the prompt
        self.assertEqual(call_args[0], "Test prompt with value")

        # Check the kwargs
        self.assertEqual(call_args[1], additional_kwargs)

        # Check the response
        self.assertEqual(call_args[2], output_model)

    def test_chat_supporting_structured_outputs(self):
        # Create a dummy Pydantic model as output schema
        class StructuredOutputModel(BaseModel):
            field1: str
            field2: int

        # Mock the structured response from the chat method
        structured_response = StructuredOutputModel(field1="direct", field2=123)
        self.llm_interface.support_structured_outputs = True

        # Replace it with an non ollama client
        self.mock_client = Mock()
        self.llm_interface.client = self.mock_client
        self.mock_client.chat.return_value = {
            "message": {"content": structured_response}
        }

        # Performing the test
        response = self.llm_interface.generate_pydantic(
            prompt_template="Dummy prompt",
            output_schema=StructuredOutputModel,
            system=self.system,
        )

        # Assertions to ensure the response is directly the structured output
        self.assertEqual(response, structured_response)

        # Ensure chat was called once with expected messages
        self.mock_client.chat.assert_called_once()

        # Check the message format
        expected_messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": "Dummy prompt"},
        ]
        call_args = self.mock_client.chat.call_args[1]
        self.assertEqual(call_args["messages"], expected_messages)
        self.assertEqual(call_args["response_schema"], StructuredOutputModel)

    def test_chat_without_system_prompt(self):
        # Create a dummy Pydantic model as output schema
        class StructuredOutputModel(BaseModel):
            field1: str
            field2: int

        # Mock the structured response from the chat method
        structured_response = StructuredOutputModel(field1="direct", field2=123)
        self.llm_interface.support_structured_outputs = True
        self.llm_interface.support_system_prompt = (
            False  # Simulate model without system prompt support
        )

        # Replace it with an non ollama client
        self.mock_client = Mock()
        self.llm_interface.client = self.mock_client
        self.mock_client.chat.return_value = {
            "message": {"content": structured_response}
        }

        # Performing the test
        response = self.llm_interface.generate_pydantic(
            prompt_template="Dummy prompt",
            output_schema=StructuredOutputModel,
            system=self.system,
        )

        # Assertions to ensure the response is directly the structured output
        self.assertEqual(response, structured_response)

        # Ensure chat was called once with expected messages
        self.mock_client.chat.assert_called_once()

        # Check the message format
        expected_messages = [
            {"role": "user", "content": "Dummy prompt"},
        ]
        call_args = self.mock_client.chat.call_args[1]
        self.assertEqual(call_args["messages"], expected_messages)
        self.assertEqual(call_args["response_schema"], StructuredOutputModel)

    def test_generate_pydantic_without_json_mode(self):
        # Create a dummy Pydantic model as output schema
        class StructuredOutputModel(BaseModel):
            field1: str
            field2: int

        # Mock the response from the chat method
        raw_response = '{"field1": "direct", "field2": 123}'
        stripped_response = '{"field1": "direct", "field2": 123}'  # Assuming the stripped response is the same for simplicity
        self.llm_interface.support_json_mode = (
            False  # Simulate model without JSON mode support
        )

        self.mock_client.chat.return_value = {"message": {"content": raw_response}}

        with patch.object(
            self.llm_interface,
            "_strip_text_from_json_response",
            return_value=stripped_response,
        ) as mock_strip:
            # Performing the test
            response = self.llm_interface.generate_pydantic(
                prompt_template="Dummy prompt",
                output_schema=StructuredOutputModel,
                system=self.system,
            )

            # Assertions to ensure the response is correctly parsed
            expected_response = StructuredOutputModel(field1="direct", field2=123)
            self.assertEqual(response, expected_response)

            # Ensure _strip_text_from_json_response was called
            mock_strip.assert_called_once_with(raw_response)

    def test_chat_with_tool_calls(self):
        # Create a mock tool
        mock_tool = mock_function

        # Setup the mock responses for multiple interactions
        self.mock_client.chat.side_effect = [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "mock_tool",
                                "arguments": {"param1": "test", "param2": 42},
                            },
                        }
                    ],
                }
            },
            {"message": {"content": "Final response after tool execution"}},
        ]

        # Make the chat request with tool
        response = self.llm_interface.chat(
            messages=[{"role": "user", "content": "Use the mock tool"}],
            tools=[mock_tool],
        )

        # Verify the chat method was called twice
        self.assertEqual(self.mock_client.chat.call_count, 2)

        # Check the final response
        self.assertEqual(response, "Final response after tool execution")

        # Verify the messages passed in the second call include tool execution results
        second_call_messages = self.mock_client.chat.call_args_list[1][1]["messages"]
        self.assertEqual(
            len(second_call_messages), 3
        )  # Original + tool call + tool result
        self.assertEqual(second_call_messages[0]["role"], "user")
        self.assertEqual(second_call_messages[1]["role"], "assistant")
        self.assertEqual(second_call_messages[2]["role"], "tool")
        self.assertEqual(second_call_messages[2]["name"], "mock_tool")

    def test_chat_with_multiple_tool_calls(self):
        mock_tool = mock_function

        # Setup mock responses for multiple tool calls
        self.mock_client.chat.side_effect = [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "mock_tool",
                                "arguments": {"param1": "first", "param2": 1},
                            },
                        }
                    ],
                }
            },
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "mock_tool",
                                "arguments": {"param1": "second", "param2": 2},
                            },
                        }
                    ],
                }
            },
            {"message": {"content": "Final response after multiple tool executions"}},
        ]

        response = self.llm_interface.chat(
            messages=[{"role": "user", "content": "Use the mock tool twice"}],
            tools=[mock_tool],
        )

        # Verify chat was called three times
        self.assertEqual(self.mock_client.chat.call_count, 3)

        # Check final response
        self.assertEqual(response, "Final response after multiple tool executions")

        # Verify the complete conversation flow
        final_call_messages = self.mock_client.chat.call_args_list[2][1]["messages"]
        self.assertEqual(
            len(final_call_messages), 5
        )  # Original + 2 tool calls + 2 results

    def test_chat_with_invalid_tool(self):
        mock_tool = mock_function

        # Setup mock response with invalid tool name
        self.mock_client.chat.return_value = {
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "invalid_tool",
                            "arguments": {"param1": "test", "param2": 42},
                        },
                    }
                ],
            }
        }

        # Make the chat request
        with self.assertLogs(level="ERROR") as log:
            _ = self.llm_interface.chat(
                messages=[{"role": "user", "content": "Use an invalid tool"}],
                tools=[mock_tool],
            )

            # Verify error was logged
            self.assertTrue(
                any(
                    "Tool 'invalid_tool' not found" in message for message in log.output
                )
            )

    def test_chat_with_tool_execution_error(self):
        # Create a mock tool that raises an exception
        @tool(name="mock_tool")
        def error_execute(self, **kwargs) -> str:
            raise Exception("Tool execution failed")

        error_tool = error_execute

        # Setup mock response
        self.mock_client.chat.return_value = {
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "mock_tool",
                            "arguments": {"param1": "test", "param2": 42},
                        },
                    }
                ],
            }
        }

        # Make the chat request
        with self.assertLogs(level="ERROR") as log:
            _ = self.llm_interface.chat(
                messages=[{"role": "user", "content": "Use the error tool"}],
                tools=[error_tool],
            )

            # Verify error was logged
            self.assertTrue(
                any("Tool execution failed" in message for message in log.output)
            )

    def test_chat_with_ollama_structured_output(self):
        # Create a LLMInterface that supports structured outputs
        llm = LLMInterface(
            model_name="llama3.1",
            client=self.mock_client,
            support_structured_outputs=True,
            support_json_mode=True,
        )
        llm.disk_cache = MockCache()

        class TestStructure(BaseModel):
            field1: str
            field2: int

        # Mock the response to return a valid JSON string
        self.mock_client.chat.return_value = {
            "message": {"content": '{"field1": "test", "field2": 42}'}
        }

        # Test the chat method with structured output
        response = llm.generate_pydantic(
            prompt_template="Test prompt",
            output_schema=TestStructure,
            system="Test system",
        )

        # Verify the response
        self.assertIsInstance(response, TestStructure)
        self.assertEqual(response.field1, "test")
        self.assertEqual(response.field2, 42)

        # Verify that the format parameter was passed correctly
        chat_kwargs = self.mock_client.chat.call_args[1]
        self.assertEqual(chat_kwargs["format"], TestStructure.model_json_schema())

    def test_create_prompt_hash(self):
        # Test that different inputs create different hashes
        base_hash = self.llm_interface._create_prompt_hash(
            model_name="model1",
            message_content="test message",
            tool_content="test tool",
            temperature=0.7,
        )

        # Different model name should create different hash
        different_model_hash = self.llm_interface._create_prompt_hash(
            model_name="model2",
            message_content="test message",
            tool_content="test tool",
            temperature=0.7,
        )
        self.assertNotEqual(base_hash, different_model_hash)

        # Different temperature should create different hash
        different_temp_hash = self.llm_interface._create_prompt_hash(
            model_name="model1",
            message_content="test message",
            tool_content="test tool",
            temperature=0.8,
        )
        self.assertNotEqual(base_hash, different_temp_hash)

        # Different message should create different hash
        different_message_hash = self.llm_interface._create_prompt_hash(
            model_name="model1",
            message_content="different message",
            tool_content="test tool",
            temperature=0.7,
        )
        self.assertNotEqual(base_hash, different_message_hash)

        # Different tool content should create different hash
        different_tool_hash = self.llm_interface._create_prompt_hash(
            model_name="model1",
            message_content="test message",
            tool_content="different tool",
            temperature=0.7,
        )
        self.assertNotEqual(base_hash, different_tool_hash)

        # Identical inputs should create identical hash
        identical_hash = self.llm_interface._create_prompt_hash(
            model_name="model1",
            message_content="test message",
            tool_content="test tool",
            temperature=0.7,
        )
        self.assertEqual(base_hash, identical_hash)

        # No temperature should be different from with temperature
        no_temp_hash = self.llm_interface._create_prompt_hash(
            model_name="model1",
            message_content="test message",
            tool_content="test tool",
        )
        self.assertNotEqual(base_hash, no_temp_hash)

        self.assertEqual(
            6,
            len(
                set(
                    [
                        base_hash,
                        different_model_hash,
                        different_temp_hash,
                        different_message_hash,
                        different_tool_hash,
                        identical_hash,
                        no_temp_hash,
                    ]
                )
            ),
        )

    def test_inject_thinking_if_needed_adds_field(self):
        # Test model without thinking field
        class TestModel(BaseModel):
            field1: str
            field2: int

        # Get new model with thinking field
        new_model = self.llm_interface._inject_thinking_if_needed(TestModel)

        # Check that the new model has the thinking field
        self.assertIn("thinking", new_model.model_fields)
        self.assertEqual(new_model.model_fields["thinking"].annotation, str)
        self.assertTrue(new_model.model_fields["thinking"].is_required())

        # Check that original fields are preserved
        self.assertIn("field1", new_model.model_fields)
        self.assertIn("field2", new_model.model_fields)

        # Verify we can create an instance with the new model
        instance = new_model(field1="test", field2=42, thinking="processing...")
        self.assertEqual(instance.thinking, "processing...")
        self.assertEqual(instance.field1, "test")
        self.assertEqual(instance.field2, 42)

    def test_inject_thinking_if_needed_preserves_existing(self):
        # Test model that already has thinking field
        class TestModelWithThinking(BaseModel):
            field1: str
            thinking: str

        # Should return the same model if thinking field exists
        same_model = self.llm_interface._inject_thinking_if_needed(
            TestModelWithThinking
        )

        # Verify it's the same class
        self.assertEqual(same_model, TestModelWithThinking)

        # Verify we can still create an instance
        instance = same_model(field1="test", thinking="already thinking")
        self.assertEqual(instance.thinking, "already thinking")
        self.assertEqual(instance.field1, "test")

    def test_inject_thinking_if_needed_inheritance(self):
        # Test that inheritance works correctly
        class BaseTestModel(BaseModel):
            base_field: str

        class DerivedModel(BaseTestModel):
            derived_field: int

        # Get new model with thinking field
        new_model = self.llm_interface._inject_thinking_if_needed(DerivedModel)

        # Check that all fields are present including inherited ones
        self.assertIn("thinking", new_model.model_fields)
        self.assertIn("base_field", new_model.model_fields)
        self.assertIn("derived_field", new_model.model_fields)

        # Verify we can create an instance with all fields
        instance = new_model(
            base_field="base", derived_field=42, thinking="processing..."
        )
        self.assertEqual(instance.base_field, "base")
        self.assertEqual(instance.derived_field, 42)
        self.assertEqual(instance.thinking, "processing...")

    def test_inject_thinking_if_needed_field_order(self):
        # Test model without thinking field
        class TestModel(BaseModel):
            field1: str
            field2: int

        # Get new model with thinking field
        new_model = self.llm_interface._inject_thinking_if_needed(TestModel)

        # Get list of field names in order
        field_names = list(new_model.model_fields.keys())
        print(field_names)

        # Verify thinking is the first field
        self.assertEqual(field_names[0], "thinking")

        # Verify other fields follow
        self.assertEqual(field_names[1:], ["field1", "field2"])

        # Verify we can still create an instance and all fields work
        instance = new_model(thinking="processing...", field1="test", field2=42)
        self.assertEqual(
            instance.model_dump_json(),
            '{"thinking":"processing...","field1":"test","field2":42}',
        )

    def test_inject_thinking_if_needed_preserves_field_order(self):
        # Test model with specific field order
        class OrderedTestModel(BaseModel):
            zfirst: str = Field(description="This should be first")
            second: int = Field(description="This should be second")
            third: bool = Field(description="This should be third")

        # Verify original model field order
        original_fields = list(OrderedTestModel.model_fields.keys())
        self.assertEqual(original_fields, ["zfirst", "second", "third"])

        # Get new model with thinking field
        new_model = self.llm_interface._inject_thinking_if_needed(OrderedTestModel)

        # Verify thinking is first, followed by original fields in order
        new_fields = list(new_model.model_fields.keys())
        self.assertEqual(new_fields, ["thinking", "zfirst", "second", "third"])

        # Create an instance and verify JSON serialization order
        instance = new_model(
            thinking="processing...", zfirst="test", second=42, third=True
        )
        self.assertEqual(
            instance.model_dump_json(),
            '{"thinking":"processing...","zfirst":"test","second":42,"third":true}',
        )

    def test_generate_pydantic_with_thinking_injection(self):
        # Create a LLMInterface that requires thinking
        llm = LLMInterface(
            model_name="llama2",
            client=self.mock_client,
            requires_thinking=True,
            support_structured_outputs=True,
        )
        llm.disk_cache = MockCache()

        # Create a simple model without thinking field
        class SimpleResponse(BaseModel):
            answer: str
            confidence: float

        # Mock a response that includes thinking
        mock_response = {
            "thinking": "Let me consider this carefully...",
            "answer": "Yes",
            "confidence": 0.95,
        }
        self.mock_client.chat.return_value = {
            "message": {"content": json.dumps(mock_response)}
        }

        # Generate response
        response = llm.generate_pydantic(
            prompt_template="Should we proceed?",
            output_schema=SimpleResponse,
            system="Be thoughtful",
        )

        # Verify that the response is of the original type (without thinking)
        self.assertIsInstance(response, SimpleResponse)

        # Verify the thinking field is not present in the final response
        self.assertFalse(hasattr(response, "thinking"))

        # Verify other fields are preserved
        self.assertEqual(response.answer, "Yes")
        self.assertEqual(response.confidence, 0.95)

        # Verify that the schema passed to the model included thinking
        chat_kwargs = self.mock_client.chat.call_args[1]
        schema = chat_kwargs["format"]
        self.assertIn("thinking", schema["properties"])
        self.assertIn("answer", schema["properties"])
        self.assertIn("confidence", schema["properties"])

    def test_generate_pydantic_with_extra_validation(self):
        # Create a dummy Pydantic model as output schema
        class ValidationTestModel(BaseModel):
            value: int
            description: str

        # Create an extra validation function that checks if value is positive
        def validate_positive_value(model: ValidationTestModel) -> Optional[str]:
            if model.value <= 0:
                return "Value must be positive"
            return None

        # Mock responses: first an invalid response, then a valid one
        self.mock_client.chat.side_effect = [
            {"message": {"content": '{"value": -1, "description": "negative"}'}},
            {"message": {"content": '{"value": 42, "description": "positive"}'}},
        ]

        # Generate response with extra validation
        response = self.llm_interface.generate_pydantic(
            prompt_template="Generate a value",
            output_schema=ValidationTestModel,
            system="Be helpful",
            extra_validation=validate_positive_value,
        )

        # Verify that the final response passes validation
        self.assertIsNotNone(response)
        self.assertEqual(response.value, 42)
        self.assertEqual(response.description, "positive")

        # Verify that chat was called twice due to validation failure
        self.assertEqual(self.mock_client.chat.call_count, 2)

        # Check that the second call included the validation error message
        second_call_messages = self.mock_client.chat.call_args_list[1][1]["messages"]
        self.assertIn("Value must be positive", second_call_messages[-1]["content"])

    def test_generate_pydantic_with_model_error(self):
        # Create a dummy Pydantic model as output schema
        class ErrorTestModel(BaseModel):
            value: str

        # Mock responses: first an error, then a valid response
        self.mock_client.chat.side_effect = [
            {"error": "Model overloaded"},
            {"message": {"content": '{"value": "success"}'}},
        ]

        # Generate response
        response = self.llm_interface.generate_pydantic(
            prompt_template="Test prompt",
            output_schema=ErrorTestModel,
            system="Test system",
        )

        # Verify that we got a valid response after retry
        self.assertIsNotNone(response)
        self.assertEqual(response.value, "success")

        # Verify chat was called twice
        self.assertEqual(self.mock_client.chat.call_count, 2)

        # Verify the retry message contains the error
        second_call_messages = self.mock_client.chat.call_args_list[1][1]["messages"]
        self.assertEqual(len(second_call_messages), 4)  # system + user + error + retry
        self.assertIn("Model overloaded", second_call_messages[2]["content"])
        self.assertIn("Try again", second_call_messages[3]["content"])

    def test_generate_pydantic_with_model_refusal(self):
        # Create a dummy Pydantic model as output schema
        class RefusalTestModel(BaseModel):
            value: str

        # Mock responses: first a refusal, then a valid response
        self.mock_client.chat.side_effect = [
            {"refusal": "Content policy violation"},
            {"message": {"content": '{"value": "appropriate content"}'}},
        ]

        # Generate response
        response = self.llm_interface.generate_pydantic(
            prompt_template="Test prompt",
            output_schema=RefusalTestModel,
            system="Test system",
        )

        # Verify that we got a valid response after retry
        self.assertIsNotNone(response)
        self.assertEqual(response.value, "appropriate content")

        # Verify chat was called twice
        self.assertEqual(self.mock_client.chat.call_count, 2)

        # Verify the retry message contains the refusal
        second_call_messages = self.mock_client.chat.call_args_list[1][1]["messages"]
        self.assertEqual(
            len(second_call_messages), 4
        )  # system + user + refusal + retry
        self.assertIn("Content policy violation", second_call_messages[2]["content"])
        self.assertIn("Try again", second_call_messages[3]["content"])

    def test_generate_pydantic_with_persistent_error(self):
        # Create a dummy Pydantic model as output schema
        class ErrorTestModel(BaseModel):
            value: str

        # Mock responses: three consecutive errors
        self.mock_client.chat.side_effect = [
            {"error": "Error 1"},
            {"error": "Error 2"},
            {"error": "Error 3"},
        ]

        # Generate response
        response = self.llm_interface.generate_pydantic(
            prompt_template="Test prompt",
            output_schema=ErrorTestModel,
            system="Test system",
        )

        # Verify that we got None after all retries failed
        self.assertIsNone(response)

        # Verify chat was called three times (maximum retries)
        self.assertEqual(self.mock_client.chat.call_count, 3)

        # Verify each retry attempt contained the previous error
        calls = self.mock_client.chat.call_args_list
        self.assertIn("Generic error: Error 1", calls[1][1]["messages"][2]["content"])
        self.assertIn("Generic error: Error 2", calls[2][1]["messages"][4]["content"])

    def test_chat_token_usage_tracking(self):
        # Create a new TokenUsage object to track tokens for this test
        test_token_usage = TokenUsage()

        # Setup mock response with token usage information (Ollama style)
        self.mock_client.chat.return_value = {
            "message": {"content": "Test response"},
            "prompt_eval_count": 50,
            "eval_count": 30,
        }

        # Call chat with the token_usage object
        self.llm_interface.chat(
            messages=[{"role": "user", "content": "Test prompt"}],
            token_usage=test_token_usage,
        )

        # Verify token usage was updated correctly
        self.assertEqual(test_token_usage.prompt_tokens, 50)
        self.assertEqual(test_token_usage.completion_tokens, 30)
        self.assertEqual(test_token_usage.total_tokens, 80)  # Should add them up

        # Also verify the default token usage in the interface was not affected
        self.assertEqual(self.llm_interface.token_usage.prompt_tokens, 0)
        self.assertEqual(self.llm_interface.token_usage.completion_tokens, 0)

    def test_chat_token_usage_openai_style(self):
        # Replace the mock client with one that returns OpenAI style responses
        openai_mock_client = Mock()
        self.llm_interface.client = openai_mock_client

        # Setup mock response with OpenAI-style token usage
        openai_mock_client.chat.return_value = {
            "message": {"content": "Test response"},
            "usage": {"prompt_tokens": 25, "completion_tokens": 15, "total_tokens": 40},
        }

        # Call chat (using default token_usage)
        self.llm_interface.chat(messages=[{"role": "user", "content": "Test prompt"}])

        # Verify token usage was updated correctly
        self.assertEqual(self.llm_interface.token_usage.prompt_tokens, 25)
        self.assertEqual(self.llm_interface.token_usage.completion_tokens, 15)
        self.assertEqual(self.llm_interface.token_usage.total_tokens, 40)

    def test_generate_pydantic_token_usage(self):
        # Create a fresh TokenUsage object
        test_token_usage = TokenUsage()

        # Create a simple Pydantic model for testing
        class SimpleModel(BaseModel):
            value: str

        # Setup mock response with token usage information
        self.mock_client.chat.return_value = {
            "message": {"content": '{"value": "test"}'},
            "prompt_eval_count": 75,
            "eval_count": 25,
        }

        # Call generate_pydantic with custom token_usage
        response = self.llm_interface.generate_pydantic(
            prompt_template="Test prompt",
            output_schema=SimpleModel,
            token_usage=test_token_usage,
        )

        # Verify token usage was updated
        self.assertEqual(test_token_usage.prompt_tokens, 75)
        self.assertEqual(test_token_usage.completion_tokens, 25)
        self.assertEqual(test_token_usage.total_tokens, 100)

        # Verify response was correctly parsed
        self.assertIsInstance(response, SimpleModel)
        self.assertEqual(response.value, "test")

    def test_generate_pydantic_multiple_iterations_token_usage(self):
        # Create a fresh TokenUsage object
        test_token_usage = TokenUsage()

        # Create a simple Pydantic model for testing
        class SimpleModel(BaseModel):
            value: str

        # Setup sequential mock responses - first invalid, then valid
        self.mock_client.chat.side_effect = [
            # First call - invalid response
            {
                "message": {"content": "Not valid JSON"},
                "prompt_eval_count": 50,
                "eval_count": 10,
            },
            # Second call - valid response
            {
                "message": {"content": '{"value": "test"}'},
                "prompt_eval_count": 100,
                "eval_count": 20,
            },
        ]

        # Call generate_pydantic
        response = self.llm_interface.generate_pydantic(
            prompt_template="Test prompt",
            output_schema=SimpleModel,
            token_usage=test_token_usage,
        )

        # Verify token usage accumulates across both calls
        self.assertEqual(test_token_usage.prompt_tokens, 150)  # 50 + 100
        self.assertEqual(test_token_usage.completion_tokens, 30)  # 10 + 20
        self.assertEqual(test_token_usage.total_tokens, 180)  # 150 + 30

        # Verify response is from the second (valid) call
        self.assertEqual(response.value, "test")

    def test_token_usage_with_cached_response(self):
        # Create a fresh TokenUsage object
        test_token_usage = TokenUsage()

        # Use the mock cache
        self.llm_interface.disk_cache = MockCache()

        # Call chat first time - should make actual call
        self.mock_client.chat.return_value = {
            "message": {"content": "Test response"},
            "prompt_eval_count": 50,
            "eval_count": 30,
        }

        response1 = self.llm_interface.chat(
            messages=[{"role": "user", "content": "Cached prompt"}],
            token_usage=test_token_usage,
        )

        # Reset mock to verify no more calls are made
        self.mock_client.chat.reset_mock()

        # Call again with same prompt - should use cache
        response2 = self.llm_interface.chat(
            messages=[{"role": "user", "content": "Cached prompt"}],
            token_usage=test_token_usage,
        )

        # Verify client.chat wasn't called second time
        self.mock_client.chat.assert_not_called()

        # Verify token usage shows only one set of tokens (not doubled)
        self.assertEqual(test_token_usage.prompt_tokens, 50)
        self.assertEqual(test_token_usage.completion_tokens, 30)
        self.assertEqual(test_token_usage.total_tokens, 80)

        # Verify responses match
        self.assertEqual(response1, response2)

    def test_chat_with_timeout_retry_logic(self):
        # Create a fresh LLMInterface with specific retry parameters
        llm = LLMInterface(
            model_name="test_model",
            client=self.mock_client,
            max_retries=2,
            retry_delay=0.1,  # Short delay for testing
        )
        llm.disk_cache = MockCache()

        # Setup mock responses - two timeouts followed by a success
        self.mock_client.chat.side_effect = [
            {
                "error": "The request timed out.",
                "error_type": errors.TIMEOUT,
                "content": None,
                "done": False,
                "usage": None,
            },
            {
                "error": "The request timed out.",
                "error_type": errors.TIMEOUT,
                "content": None,
                "done": False,
                "usage": None,
            },
            {"message": {"content": "Success after retries"}},
        ]

        # Call chat method which should trigger retries
        with patch("llm_interface.llm_interface.time.sleep") as mock_sleep:
            response = llm.chat(messages=[{"role": "user", "content": "Test message"}])

            # Verify sleep was called for each retry with increasing delay
            self.assertEqual(mock_sleep.call_count, 2)
            mock_sleep.assert_any_call(0.1)  # First retry should use base delay
            mock_sleep.assert_any_call(0.2)  # Second retry should use doubled delay

        # Verify the chat method was called the expected number of times (initial + 2 retries)
        self.assertEqual(self.mock_client.chat.call_count, 3)

        # Verify we got the successful response after retries
        self.assertEqual(response, "Success after retries")

    def test_chat_with_connection_error_retry(self):
        # Create a fresh LLMInterface with specific retry parameters
        llm = LLMInterface(
            model_name="test_model",
            client=self.mock_client,
            max_retries=1,
            retry_delay=0.1,  # Short delay for testing
        )
        llm.disk_cache = MockCache()

        # Setup mock responses - connection error followed by a success
        self.mock_client.chat.side_effect = [
            {
                "error": "Connection refused",
                "error_type": errors.CONNECTION,
                "content": None,
                "done": False,
                "usage": None,
            },
            {"message": {"content": "Success after connection retry"}},
        ]

        # Call chat method which should trigger a retry
        with patch("llm_interface.llm_interface.time.sleep") as mock_sleep:
            response = llm.chat(messages=[{"role": "user", "content": "Test message"}])

            # Verify sleep was called once
            mock_sleep.assert_called_once_with(0.1)

        # Verify the chat method was called twice (initial + 1 retry)
        self.assertEqual(self.mock_client.chat.call_count, 2)

        # Verify we got the successful response after the retry
        self.assertEqual(response, "Success after connection retry")

    def test_chat_with_exhausted_retries(self):
        # Create a fresh LLMInterface with specific retry parameters
        llm = LLMInterface(
            model_name="test_model",
            client=self.mock_client,
            max_retries=2,
            retry_delay=0.1,  # Short delay for testing
        )
        llm.disk_cache = MockCache()

        # Setup mock responses - all timeouts (exceeding max_retries)
        self.mock_client.chat.return_value = {
            "error": "The request timed out.",
            "error_type": errors.TIMEOUT,
            "content": None,
            "done": False,
            "usage": None,
        }

        # Call chat method which should trigger retries and then fail
        with patch("llm_interface.llm_interface.time.sleep") as mock_sleep:
            with self.assertRaises(ModelError) as context:
                llm.chat(messages=[{"role": "user", "content": "Test message"}])

            # Verify sleep was called for each retry
            self.assertEqual(mock_sleep.call_count, 2)

        # Verify the chat method was called the expected number of times (initial + max_retries)
        self.assertEqual(self.mock_client.chat.call_count, 3)

        # Verify the error message contains timeout information
        self.assertIn("Generic error: The request timed out.", str(context.exception))

    def test_chat_allow_json_mode(self):
        # Test when allow_json_mode is True (default)
        self.llm_interface.support_json_mode = True
        self.mock_client.chat.return_value = {
            "message": {"content": '{"key": "value"}'}
        }

        self.llm_interface.chat(
            messages=[{"role": "user", "content": "Get JSON"}],
            allow_json_mode=True,
        )

        # Verify format="json" was passed
        call_args = self.mock_client.chat.call_args[1]
        self.assertEqual(call_args.get("format"), "json")

        # Reset mock for next call
        self.mock_client.chat.reset_mock()

        # Test when allow_json_mode is False
        self.llm_interface.chat(
            messages=[{"role": "user", "content": "Get text"}],
            allow_json_mode=False,
        )

        # Verify format="json" was NOT passed
        call_args = self.mock_client.chat.call_args[1]
        self.assertNotIn("format", call_args)

        # Reset mock for next call
        self.mock_client.chat.reset_mock()

        # Test when response_schema is provided (should ignore allow_json_mode)
        class JsonSchema(BaseModel):
            key: str

        self.llm_interface.support_structured_outputs = True
        self.llm_interface.client = Mock()  # Use a non-Ollama mock
        self.llm_interface.client.chat.return_value = {
            "message": {"content": JsonSchema(key="value")}
        }
        self.llm_interface.generate_pydantic(
            prompt_template="Get structured JSON",
            output_schema=JsonSchema,
            allow_json_mode=False,  # This should be ignored
        )

        # Verify response_schema was passed, not format="json"
        call_args = self.llm_interface.client.chat.call_args[1]
        self.assertNotIn("format", call_args)
        self.assertEqual(call_args.get("response_schema"), JsonSchema)

        # Reset mock and settings for next call
        self.mock_client.chat.reset_mock()
        self.llm_interface.support_structured_outputs = False
        self.llm_interface.client = self.mock_client  # Restore original mock

        # Clear cache before the next call to ensure mock is called
        self.llm_interface.disk_cache = MockCache()

        # Test when support_json_mode is False
        self.llm_interface.support_json_mode = False
        self.llm_interface.chat(
            messages=[{"role": "user", "content": "Get text"}],
            allow_json_mode=True,  # This should be ignored
        )

        # Verify format="json" was NOT passed
        self.mock_client.chat.assert_called_once()  # Ensure the mock was called
        call_args = self.mock_client.chat.call_args[1]
        self.assertNotIn("format", call_args)


if __name__ == "__main__":
    unittest.main()
