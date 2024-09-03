from typing import List, Tuple
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import typer
import evaluate
import string
from datasets import load_dataset
import rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

from experiments.evaluator import ModelsEnum, LLMEvaluator  # noqa: E402

app = typer.Typer()

class ParaphrasingEvaluator(LLMEvaluator):

    def __init__(self, texts: List[str], golds: List[str], prompt, system_prompt, model: ModelsEnum, port: int, lang: str, dataset_name: str, task_name: str, output_dir_path: Path, report_path: Path):
        # call parent
        super(ParaphrasingEvaluator, self).__init__(model=model, port=port, lang=lang, dataset_name=dataset_name, task_name=task_name, output_dir_path=output_dir_path, report_path=report_path)

        assert len(texts) == len(golds), "Length of texts and golds must be the same."
        print(f"Input: {len(texts)} Texts.")
        self.texts = texts
        self.golds = golds

        assert len(prompt) > 0, "Prompt must not be empty."
        self.prompt = prompt

        assert len(system_prompt) > 0, "System prompt must not be empty."
        self.system_prompt = system_prompt.strip()

    def _parse_response(self, response: str) -> str:
        result = ""

        for line in response.strip().split("\n"):
            if not line.strip():
                continue
            if ":" not in line:
                continue

            splitted_line = line.split(":")
            if len(splitted_line) != 2:
                continue

            result = splitted_line[1].strip()

        return result

    def _prompt_ollama(self, text: str) -> Tuple[str, str]:
        response = self.client.chat(model=self.model, messages=[
            {
                'role': 'system',
                'content': self.system_prompt,
            },
            {
                'role': 'user',
                'content': self.prompt.format(text).strip(),
            },
        ])
        
        message = response["message"]["content"]
        prediction = self._parse_response(message)
        return message, prediction

    def _evaluate(self):
        preds = []
        messages = []   
        for text in tqdm(self.texts, desc="Evaluating"):
            message, prediction = self._prompt_ollama(text=text)

            preds.append(prediction)
            messages.append(message)

        # store the evaluation results
        pd.DataFrame({
            "Gold": self.golds,
            "Prediction": preds,
            "Message": messages
        }).to_parquet(self.output_file_path.with_suffix(".parquet"))

    def _remove_punctuation(self, text: str) -> str:
        return text.translate(str.maketrans("", "", string.punctuation))

    def _transform_to_squad(self, text: List[str], is_gold: bool) -> List[str]:
        result = []
        for idx, sentence in enumerate(text):
            if is_gold:
                result.append({'answers': {"answer_start": [0], "text": [sentence]}, "id": str(idx)})
            else:
                result.append({'prediction_text': sentence, 'id': str(idx), 'no_answer_probability': 0.0})

        return result

    def _report(self):
        # load the predictions
        df = pd.read_parquet(self.output_file_path.with_suffix(".parquet"))
        golds = [self._remove_punctuation(x) for x in df["Gold"].to_list()]
        preds = [self._remove_punctuation(x) for x in df["Prediction"].to_list()]        

        # rouge eval
        rouge = evaluate.load("rouge")
        rouge_result = rouge.compute(predictions=preds, references=golds)
        print("Rouge:")
        print(rouge_result)
        if rouge_result is None:
            raise ValueError("Rouge result is None.")

        # EM/F1 eval
        squad_v2_metric = evaluate.load("squad_v2")
        gold_transformed = self._transform_to_squad(golds, is_gold=True)
        pred_transformed = self._transform_to_squad(preds, is_gold=False)
        squad_result = squad_v2_metric.compute(references=gold_transformed, predictions=pred_transformed)
        print("EM/F1:")
        print(squad_result)
        if squad_result is None:
            raise ValueError("Squad result is None.")

        # METEOR eval
        meteor = evaluate.load("meteor")
        meteor_result = meteor.compute(predictions=preds, references=golds)
        print("Meteor:")
        print(meteor_result)
        if meteor_result is None:
            raise ValueError("Meteor result is None.")

        # write report
        self._add_results_to_report({
            "Rouge 1": round(rouge_result["rouge1"] * 100, 2),
            "Rouge 2": round(rouge_result["rouge2"] * 100, 2),
            "Rouge L": round(rouge_result["rougeL"] * 100, 2),
            "Rouge Lsum": round(rouge_result["rougeLsum"] * 100, 2),
            "Exact Match": round(squad_result["exact"], 2),
            "F1": round(squad_result["f1"], 2),
            "METEOR": round(meteor_result["meteor"] * 100, 2),
        })


@app.command()
def disflqa(model: ModelsEnum, port: int, report_only: bool = False):
    # load dataset
    df = pd.read_parquet("datasets/disfl-qa/disfl_qa_test.parquet")

    texts = df["disfluent"].to_list()
    golds = df["original"].to_list()

    # prompts
    system_prompt = "You are a system to support the analysis of large amounts of text. You will assist the user by rephrasing the provided texts. You will always answer in the required format and use no other formatting than expected by the user!"
    user_prompt = """
I have a noisy, disfluent text. I need you to remove all disfluencies from the text below. Keep the text as close to the original as possible, but make sure it is fluent to read.

Text: {}

Respond in the following format:
Fluent text: <the corrected text>

e.g.
Fluent text: What time is it?.

Remember, you MUST keep to the original text as much as possible, do not generate new content!
"""

    # start evaluator
    evaluator = ParaphrasingEvaluator(
        model=model,
        port=port,
        lang="en",
        task_name="disfluency-correction",
        dataset_name="DisflQA",
        output_dir_path=Path("experiments/paraphrasing/results/"),
        report_path=Path("experiments/paraphrasing/report.csv"),
        texts=texts,
        golds=golds,
        prompt=user_prompt,
        system_prompt=system_prompt
    )
    evaluator.start(report_only=report_only)

@app.command()
def disco_en(model: ModelsEnum, port: int, report_only: bool = False):
    # load dataset
    df = pd.read_parquet("datasets/disco/disco_en.parquet")

    texts = df["Disfluent Sentence"].to_list()
    golds = df["Fluent Sentence"].to_list()

    # prompts
    system_prompt = "You are a system to support the analysis of large amounts of text. You will assist the user by rephrasing the provided texts. You will always answer in the required format and use no other formatting than expected by the user!"
    user_prompt = """
I have a noisy, disfluent text. I need you to remove all disfluencies from the text below. Keep the text as close to the original as possible, but make sure it is fluent to read.

{}

Respond in the following format:
Fluent text: <the corrected text>

e.g.
Fluent text: This picture looks great.

Remember, you MUST keep to the original text as much as possible, do not generate new content!
"""

    # start evaluator
    evaluator = ParaphrasingEvaluator(
        model=model,
        port=port,
        lang="en",
        task_name="disfluency-correction",
        dataset_name="DISCO-en",
        output_dir_path=Path("experiments/paraphrasing/results/"),
        report_path=Path("experiments/paraphrasing/report.csv"),
        texts=texts,
        golds=golds,
        prompt=user_prompt,
        system_prompt=system_prompt
    )
    evaluator.start(report_only=report_only)
    
@app.command()
def disco_de(model: ModelsEnum, port: int, report_only: bool = False):
    # load dataset
    df = pd.read_parquet("datasets/disco/disco_de.parquet")

    texts = df["Disfluent Sentence"].to_list()
    golds = df["Fluent Sentence"].to_list()

    # prompts
    system_prompt = "Du bist ein System zur Unterstützung der Analyse großer Textmengen. Du wirst dem Benutzer helfen, die bereitgestellten Texte umzuformulieren. Du wirst immer in dem erforderlichen Format antworten und die vom Benutzer erwartete Formatierung verwenden!"
    user_prompt = """
Ich habe hier einen nicht flüssig geschriebenen Text. Ich möchte, das du alle Unstimmigkeiten aus dem Text unten entfernst. Halte den Text so nah wie möglich am Original, aber stelle sicher, dass er flüssig zu lesen ist.

{}

Antworte im folgenden Format:
Text: <flüssiger Text>

z.B.
Text: Dieses Bild sieht fantastisch aus.

Denke daran, du MUSST dich so nah wie möglich an den Originaltext halten, generiere keinen neuen Inhalt!
"""

    # start evaluator
    evaluator = ParaphrasingEvaluator(
        model=model,
        port=port,
        lang="de",
        task_name="disfluency-correction",
        dataset_name="DISCO-de",
        output_dir_path=Path("experiments/paraphrasing/results/"),
        report_path=Path("experiments/paraphrasing/report.csv"),
        texts=texts,
        golds=golds,
        prompt=user_prompt,
        system_prompt=system_prompt
    )
    evaluator.start(report_only=report_only)

@app.command()
def cnndm(model: ModelsEnum, port: int, report_only: bool = False):
    # load dataset
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")["test"]

    texts = dataset["article"]
    golds = dataset["highlights"]

    # prompts
    system_prompt = "You are a system to support the analysis of large amounts of text. You will assist the user by summarizing the provided texts. You will always answer in the required format and use no other formatting than expected by the user!"
    user_prompt = """
Please write a concise summary of the text below, highlighting the most important information. Try to use about 50 - 60 words only.

{}

Respond in the following format:
Summary: <summarized text>

e.g.
Summary: Theia was hit by a car ...

Remember, you MUST summarize the original text, do not generate new facts!
"""

    # start evaluator
    evaluator = ParaphrasingEvaluator(
        model=model,
        port=port,
        lang="en",
        task_name="summarization",
        dataset_name="CNNDM",
        output_dir_path=Path("experiments/paraphrasing/results/"),
        report_path=Path("experiments/paraphrasing/report.csv"),
        texts=texts,
        golds=golds,
        prompt=user_prompt,
        system_prompt=system_prompt
    )
    evaluator.start(report_only=report_only)

@app.command()
def mlsum(model: ModelsEnum, port: int, report_only: bool = False):
    # load dataset
    dataset = load_dataset("mlsum", "de")["test"]

    texts = dataset["text"]
    golds = dataset["summary"]

    # prompts
    system_prompt = "Du bist ein System zur Unterstützung bei der Analyse großer Textmengen. Du wirst dem Benutzer helfen, die bereitgestellten Texte zusammenzufassen. Du wirst immer in dem erforderlichen Format antworten und die vom Benutzer erwartete Formatierung verwenden!"
    user_prompt = """
Bitte verfasse eine kurze und prägnante Zusammenfassung des untenstehenden Textes, in der die wichtigsten Informationen hervorgehoben werden. Verwende möglichst nur etwa 25 - 35 Wörter!

{}

Antworte im folgenden Format:
Zusammenfassung: <zusammengefasster Text>

z.B.
Zusammenfassung: Theia wurde von einem Auto angefahren ...

Denke daran, du MUSST dich an den Originaltext halten, generiere keine neuen Fakten!
"""

    # start evaluator
    evaluator = ParaphrasingEvaluator(
        model=model,
        port=port,
        lang="de",
        task_name="summarization",
        dataset_name="MLSUM",
        output_dir_path=Path("experiments/paraphrasing/results/"),
        report_path=Path("experiments/paraphrasing/report.csv"),
        texts=texts,
        golds=golds,
        prompt=user_prompt,
        system_prompt=system_prompt
    )
    evaluator.start(report_only=report_only)
    

if __name__ == "__main__":
    app()
