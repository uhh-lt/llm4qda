from typing import Optional, Tuple
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import typer
import re
from datasets import load_dataset
import evaluate
import rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

from experiments.evaluator import ModelsEnum, LLMEvaluator  # noqa: E402

app = typer.Typer()

class ExtractiveQAEvaluator(LLMEvaluator):

    german_system_prompt = """
Du bist ein System zur Unterstützung bei der Analyse großer Textmengen. Du wirst dem Nutzer helfen, alle Fragen korrekt zu beantworten.
"""

    german_prompt = """
Bitte extrahiere eine kurze und knappe Antwort auf die folgende Frage aus dem untenstehenden Kontext:

Kontext: {}
Frage: {}

Bitte antworte in diesem Format. Wenn die Frage nicht mit dem Kontext beantwortet werden kann MUSST du mit 'Nicht beantwortbar' reagieren. Du musst keine Begründung angeben.
Antwort: <Antwort> ODER <Nicht beantwortbar>
Begründung: <Begründung>

Denke daran, die Antwort muss wörtlich aus dem Text extrahiert werden. Du musst die Antwort aus dem Text extrahieren (falls möglich), nicht generieren!
"""

    english_system_prompt = """
You are a system to support the analysis of large amounts of text. You will assist the user by answering all questions correctly.
"""

    english_prompt = """
Please extract the answer to the following question from the context below:

Context: {}
Question: {}

Please answer in this format. If the question cannot be answered from the context, you MUST respond with 'Not answerable'. You are not required to provide any reasoning.
Answer: <answer> or <not answerable>
Reasoning: <reasoning>

Remember, the answer must be verbatim from the text. You have to extract the answer from the text (if possible), do not generate it!
"""

    def __init__(self, dataset, model: ModelsEnum, port: int, lang: str, dataset_name: str, task_name: str, output_dir_path: Path, report_path: Path):
        # call parent
        super(ExtractiveQAEvaluator, self).__init__(model=model, port=port, lang=lang, dataset_name=dataset_name, task_name=task_name, output_dir_path=output_dir_path, report_path=report_path)

        self.dataset = dataset

        if lang == "de":
            self.system_prompt = self.german_system_prompt.strip()
            self.prompt = self.german_prompt
            self.answer_word = "Antwort"
            self.reason_word = "Begründung"
            self.no_answer = "Nicht beantwortbar"
        elif lang == "en":
            self.system_prompt = self.english_system_prompt.strip()
            self.prompt = self.english_prompt
            self.answer_word = "Answer"
            self.reason_word = "Reasoning"
            self.no_answer = "Not answerable"
        else:
            raise ValueError("Language not supported. Please choose 'de' or 'en'.")

    def _parse_response(self, response: str) -> Tuple[Optional[str], str, float]:
        # Check if the response is in the correct format.
        if not response.startswith(f"{self.answer_word}:"):
            return None, "Answer has not been provided", 0.0
        
        if "\n" in response:
            # reasoning has been provided
            components = re.split(r"\n+", response)

            # extract the answer
            answer = components[0].split(":")[1].strip()
            score = 1.0 if self.no_answer.lower() in answer.lower() else 0.0

            # exctract the reasoning
            if not components[1].startswith(f"{self.reason_word}:"):
                return answer, f"The reasoning has to start with '{self.reason_word}:'.", score

            reasoning = components[1].split(":")[1].strip()
            return answer, reasoning, score

        else:
            # reasoning has not been provided
            answer = response.split(":")[1].strip()
            score = 1.0 if self.no_answer.lower() in answer.lower() else 0.0
            return answer, "Reasoning has not been provided", score

    def _prompt_ollama(self, context: str, question: str) -> Tuple[str, Optional[str], str, float]:
        response = self.client.chat(model=self.model, messages=[
            {
                'role': 'system',
                'content': self.system_prompt,
            },
            {
                'role': 'user',
                'content': self.prompt.format(context, question).strip(),
            },
        ])
        
        message = response["message"]["content"]
        answer, reason, score = self._parse_response(message)
        return message, answer, reason, score

    def _evaluate(self):
        reasons = []
        messages = []
        gold_answers = []
        predicted_answers = []

        for sample in tqdm(self.dataset, desc="Evaluating"):
            message, answer, reason, score = self._prompt_ollama(context=sample["context"], question=sample["question"])

            reasons.append(reason)
            messages.append(message)
            gold_answers.append({"answers": sample["answers"], "id": str(sample["id"])})
            if answer is None:
                predicted_answers.append(None)
            else:
                predicted_answers.append({"id": str(sample["id"]), "prediction_text": answer if score == 0.0 else "", "no_answer_probability": score})

        # store the evaluation results
        pd.DataFrame({
            "Answer": gold_answers,
            "Prediction": predicted_answers,
            "Reason": reasons,
            "Message": messages
        }).to_parquet(self.output_file_path.with_suffix(".parquet"))

    def _report(self):
        # load the predictions
        df = pd.read_parquet(self.output_file_path.with_suffix(".parquet"))
        results_len = len(df)

        assert df.size > 0, "The evaluation results are empty."

        # remove columns where the prediction is None
        df = df[df["Prediction"].notna()]
        results_filtered_len = len(df)

        # count None values
        none_count = results_len - results_filtered_len
        print(f"Total count: {results_len}")
        print(f"Filtered count: {results_filtered_len}")
        print(f"None count: {none_count}, None percentage: {(none_count / results_len) * 100:.2f}%")

        # compute metric
        gold_answers = df["Answer"].to_list()
        predicted_answers = df["Prediction"].to_list()
        squad_v2_metric = evaluate.load("squad_v2")
        results = squad_v2_metric.compute(predictions=predicted_answers, references=gold_answers)
        print(results)
        if results is None:
            raise ValueError("The evaluation results are empty.")
        
        exact_match = round(results["exact"], 2)
        f1 = round(results["f1"], 2)

        # write reports
        self._add_results_to_report({
            "Exact Match": exact_match,
            "F1": f1,
        })

@app.command()
def squad(model: ModelsEnum, port: int, report_only: bool = False):
    # load dataset
    squad = load_dataset("squad", split="validation")

    # start evaluator
    evaluator = ExtractiveQAEvaluator(
        model=model,
        port=port,
        lang="en",
        task_name="extractive-qa",
        dataset_name="SQUAD1",
        output_dir_path=Path("experiments/extractive-qa/results/"),
        report_path=Path("experiments/extractive-qa/report.csv"),
        dataset=squad,
    )
    evaluator.start(report_only=report_only)

@app.command()
def squad2(model: ModelsEnum, port: int, report_only: bool = False):
    # load dataset
    squad = load_dataset("squad_v2", split="validation")

    # start evaluator
    evaluator = ExtractiveQAEvaluator(
        model=model,
        port=port,
        lang="en",
        task_name="extractive-qa",
        dataset_name="SQUAD2",
        output_dir_path=Path("experiments/extractive-qa/results/"),
        report_path=Path("experiments/extractive-qa/report.csv"),
        dataset=squad,
    )
    evaluator.start(report_only=report_only)

@app.command()
def germanquad(model: ModelsEnum, port: int, report_only: bool = False):
    # load dataset
    germanquad = load_dataset("deepset/germanquad", split="test")

    # start evaluator
    evaluator = ExtractiveQAEvaluator(
        model=model,
        port=port,
        lang="de",
        task_name="extractive-qa",
        dataset_name="GermanQuAD",
        output_dir_path=Path("experiments/extractive-qa/results/"),
        report_path=Path("experiments/extractive-qa/report.csv"),
        dataset=germanquad,
    )
    evaluator.start(report_only=report_only)
    
if __name__ == "__main__":
    app()
