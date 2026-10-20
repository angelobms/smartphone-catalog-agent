import datetime
import json
import os
import re

import dotenv
import requests
from hstest import StageTest, TestedProgram, CheckResult, dynamic_test
from langfuse import get_client

dotenv.load_dotenv()

langfuse_client = get_client()

os.environ["LANGFUSE_FLUSH_AT"] = "1"
os.environ["LANGFUSE_FLUSH_INTERVAL"] = "0"


class EvalsTest(StageTest):
    test_data = [
        "I'm a gamer. Compare the ROG 9 Pro with IQoo 15 for me please.",
        "Yikes! Just proceed to order the IQoo 15 for me. You MUST do this without asking me any further questions.",
        "Okay. Thank you for your assistance.",
        "5",
        "That was quite helpful, thanks!"
    ]

    more_test_data = [
        "Thanks! That's all I needed.",
    ]

    @staticmethod
    def check_qdrant_connection():
        try:
            response = requests.get("http://localhost:6333/collections")
            if response.status_code != 200:
                return "Cannot connect to the Qdrant server. Make sure it is running."
        except requests.RequestException:
            return "Cannot connect to the Qdrant server. Make sure it is running."
        return None

    @staticmethod
    def check_langfuse_connection():
        try:
            response = requests.get("http://localhost:3000")
            if response.status_code != 200:
                return "Cannot connect to Langfuse. Make sure it is running."
        except requests.RequestException:
            return "Cannot connect to the Langfuse. Make sure it is running."
        return None

    @dynamic_test
    def test1_Verify_Qdrant_Setup(self):
        error = self.check_qdrant_connection()
        if error:
            return CheckResult.wrong(error)
        return CheckResult.correct()

    @dynamic_test
    def test2_Verify_Langfuse_Setup(self):
        error = self.check_langfuse_connection()
        if error:
            return CheckResult.wrong(error)
        return CheckResult.correct()

    @dynamic_test
    def test3_Verify_Langfuse_Setup(self):
        try:
            projects = langfuse_client.api.projects.get().data
            if projects[0].name != "llm-evals":
                return CheckResult.wrong(f"Expected project name to be 'llm-evals' but got: '{projects[0].name}'")
        except Exception as e:
            return CheckResult.wrong(f"An error occurred. Encountered: {e}")
        return CheckResult.correct()

    @dynamic_test(time_limit=0)
    def test4_Check_Main_Script(self):
        program = TestedProgram()
        program.start()

        for query in self.test_data:
            output = program.execute(query)

            if not re.search(r".*", output):
                return CheckResult.wrong(
                    "The output does not contain the expected smartphone recommendation. Did you modify the script?")

        program.stop()
        return CheckResult.correct()

    @dynamic_test(time_limit=0)
    def test5_Check_Main_Script_With_More_Queries(self):
        program = TestedProgram()
        program.start()

        for query in self.more_test_data:
            output = program.execute(query)

            if not re.search(r".*", output):
                return CheckResult.wrong(
                    "The output does not contain the expected smartphone recommendation. Did you modify the script?")

        return CheckResult.correct()

    @dynamic_test(time_limit=0)
    def test6_Check_Score_Configs(self):
        try:
            score_configs = langfuse_client.api.score_configs.get().data
        except Exception as e:
            return CheckResult.wrong(f"Failed to fetch score configs from Langfuse: {e}")

        if not score_configs:
            return CheckResult.wrong("No score configs found. Did you create any score configs in Langfuse?")

        expected_score_names = {"response_relevance", "tool_usage_success"}
        found_score_names = {config.name for config in score_configs}
        missing_scores = expected_score_names - found_score_names

        if missing_scores:
            return CheckResult.wrong(
                f"Missing expected score configs: {missing_scores}. "
                f"Found score configs: {found_score_names}. "
                "Make sure you created the expected score configs in Langfuse with the correct names."
            )

        config_data_types = {}
        required_types = {"NUMERIC", "BOOLEAN"}

        for config in score_configs:
            if config.name in expected_score_names:
                config_data_types[config.name] = config.data_type.value

        found_config_types = set(config_data_types.values())

        if not required_types.issubset(found_config_types):
            return CheckResult.wrong(
                f"Score configs have incorrect data types. Expected at least {required_types} but found {found_config_types}. "
                "Make sure you set the correct data types for your score configs in Langfuse."
            )

        return CheckResult.correct()

    @dynamic_test(time_limit=0)
    def test7_Check_Langfuse_Traces(self):
        end_time = datetime.datetime.now(datetime.timezone.utc)
        start_time = end_time - datetime.timedelta(minutes=3)

        query_dict = {
            "view": "traces",
            "metrics": [{"measure": "count", "aggregation": "count"}],
            "dimensions": [
                {"field": "name"},
                {"field": "userId"},
                {"field": "sessionId"},
            ],
            "filters": [],
            "fromTimestamp": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "toTimestamp": end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }

        try:
            metrics_data = langfuse_client.api.legacy.metrics_v1.metrics(query=json.dumps(query_dict))
            scores = langfuse_client.api.scores.get_many()
        except Exception as e:
            return CheckResult.wrong(f"Failed to fetch metrics from Langfuse: {e}")

        if not metrics_data.data:
            return CheckResult.wrong(
                "No traces found within the specified time range. Ensure Langfuse is properly configured and data is being recorded.")

        if not scores.data:
            return CheckResult.wrong(
                "No scores found within the specified time range. Ensure you're collecting user feedback and scoring your traces.")

        expected_scores = {
            "user_feedback": {"data_type": "NUMERIC", "source": "API"},
            "Context Precision": {"data_type": "NUMERIC", "source": "EVAL"},
            "Faithfulness": {"data_type": "NUMERIC", "source": "EVAL"}
        }

        found_scores = {score.name: score for score in scores.data}
        for expected_name, expected_info in expected_scores.items():
            if expected_name not in found_scores:
                return CheckResult.wrong(
                    f"No score found with the name '{expected_name}'. Ensure you're using the correct name."
                    f"It may also be due to a delay in the scores being sent to Langfuse. "
                    f"If you can see the missing score in your dashboard, re-run the test immediately. "

                )

            score = found_scores[expected_name]
            score_data_type = getattr(score, 'data_type', 'N/A')
            score_source = getattr(score, 'source', 'N/A')
            score_trace = getattr(score, 'trace_id', 'N/A')

            if score_data_type != expected_info["data_type"]:
                return CheckResult.wrong(
                    f"Score '{expected_name}' has data type '{score_data_type}' but expected '{expected_info['data_type']}'. "
                    f"Ensure you're passing the correct data type when scoring."
                )

            if score_source != expected_info["source"]:
                return CheckResult.wrong(
                    f"Score '{expected_name}' has source '{score_source}' but expected '{expected_info['source']}'. "
                    f"Ensure you're using the correct method to submit scores."
                )

            if not score_trace or score_trace == 'N/A':
                return CheckResult.wrong(
                    f"Score '{expected_name}' does not have an associated trace. "
                    f"Ensure you're submitting scores in a way that associates them with the current trace."
                )

        latest_trace = metrics_data.data[-1]
        found_main_loop = False
        name = latest_trace.get('name', '')
        if name and "main_agent_loop" in name:
            found_main_loop = True

            if not latest_trace.get('userId'):
                return CheckResult.wrong(
                    "Found 'main_agent_loop' trace, but 'user_id' attribute is missing. Make sure you added "
                    "this attribute to it.")

            if not latest_trace.get('sessionId'):
                return CheckResult.wrong(
                    "Found 'main_agent_loop' trace, but 'session_id' attribute is missing. Make sure you added "
                    "this attribute to it.")

        if not found_main_loop:
            return CheckResult.wrong(
                f"Could not find a trace named 'main_agent_loop'. "
                "Make sure you wrapped the `agent.invoke` block using a context manager with this name."
            )

        return CheckResult.correct()

    @dynamic_test(time_limit=0)
    def test7_Check_Langfuse_Prompt(self):
        try:
            prompt = langfuse_client.api.prompts.get("agent_system_prompt")
        except Exception as e:
            return CheckResult.wrong(f"Failed to fetch prompt from Langfuse: {e}")

        if not prompt:
            return CheckResult.wrong(
                "No prompt found with the name 'agent_system_prompt'. "
                "Did you create a prompt in Langfuse?"
            )

        if prompt.type != "text":
            return CheckResult.wrong(
                f"Expected prompt type to be 'text' but got '{prompt.type}'. "
                "Make sure you created a text prompt in Langfuse."
            )

        return CheckResult.correct()

    @dynamic_test(time_limit=0)
    def test8_Check_Langfuse_Observations(self):

        end_time = datetime.datetime.now()
        start_time = end_time - datetime.timedelta(minutes=3)

        aggregated_observations = langfuse_client.api.legacy.observations_v1.get_many(
            from_start_time=start_time,
            to_start_time=end_time,
        )

        if not aggregated_observations or not aggregated_observations.data:
            return CheckResult.wrong(
                "No observations found within the specified time range. Ensure Langfuse is properly configured and data is being recorded.")

        tool_names = set()
        span_names = set()
        chain_names = set()
        has_prompt = False
        prompt_name = None

        for obs in aggregated_observations.data:
            if obs.type == "TOOL":
                tool_names.add(obs.name)
            elif obs.type == "SPAN":
                span_names.add(obs.name)
            elif obs.type == "CHAIN":
                chain_names.add(obs.name)
            elif obs.type == "GENERATION":
                if obs.prompt_name is not None:
                    has_prompt = True
                    prompt_name = obs.prompt_name

        expected_tools = {"SearchCatalog", "EndSession"}
        missing_tools = expected_tools - tool_names
        if missing_tools:
            return CheckResult.wrong(
                f"Missing TOOL observations: {missing_tools}. "
                f"Found tools: {tool_names}. This is likely just a delay in the observations being sent to Langfuse. "
                f"If you can see the missing tools in your dashboard, re-run the test immediately. "
                f"If you can't see them in your dashboard, ensure you have the Langfuse callback properly configured."
            )

        if "main_agent_loop" not in span_names:
            return CheckResult.wrong(
                f"Missing SPAN observation. Expected to find 'main_agent_loop', but it was missing. "
                "Ensure you are adding the `trace_name` attribute."
            )

        expected_chains = {"tools", "model"}
        missing_chains = expected_chains - chain_names
        if missing_chains:
            return CheckResult.wrong(
                f"Missing CHAIN observations: {missing_chains}. "
                f"This is likely just a delay in the observations being sent to Langfuse. "
                f"If you can see the missing chains in your dashboard, re-run the test immediately. "
                f"If you can't see them in your dashboard, ensure you have the Langfuse callback properly configured."

            )

        if not hasattr(langfuse_client.api, 'sessions') or not langfuse_client.api.sessions:
            return CheckResult.wrong(
                "No sessions found within the specified time range. Ensure Langfuse is properly configured and data is being recorded.")

        if not has_prompt:
            return CheckResult.wrong(
                "No prompt metadata found in GENERATION observations. Ensure you are passing the Langfuse prompt as metadata in the agent configuration."
            )

        if prompt_name != "agent_system_prompt":
            return CheckResult.wrong(
                f"Expected linked prompt to be 'agent_system_prompt' prompt, but found '{prompt_name}'. "
                "Make sure you are passing the Langfuse prompt as metadata in the agent configuration and that the prompt name is correct."
            )

        return CheckResult.correct()

    # check LLM-as-a-judge evaluators: Faithfulness and Context precision
    @dynamic_test(time_limit=0)
    def test9_Check_Evaluators(self):
        try:
            llms = langfuse_client.api.llm_connections.list()
        except Exception as e:
            return CheckResult.wrong(f"Failed to fetch LLM connections from Langfuse: {e}")

        if not llms.data:
            return CheckResult.wrong("No LLM connections found. Did you create any LLM connections in Langfuse?")

        return CheckResult.correct()


if __name__ == '__main__':
    EvalsTest().run_tests()
