from typing import List, Tuple
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import typer
import rootutils
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from datasets import load_dataset
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

from experiments.evaluator import ModelsEnum, LLMEvaluator  # noqa: E402

app = typer.Typer()

class SpanClassificationEvaluator(LLMEvaluator):

    german_system_prompt = """
Du bist ein System zur Unterstützung bei der Analyse großer Textmengen. Du unterstützt den Nutzer, indem du die angeforderten Informationen aus dem bereitgestellten Dokument extrahierst. Du antwortest immer in dem geforderten Format und verwendest keine andere Formatierung als vom Benutzer erwartet!
"""

    german_prompt = """
Hier ist eine Liste von Entitätskategorien, die ich aus dem Text extrahieren möchte. Die Kategorien sind:
{}

Bitte extrahiere passende Entitäten aus dem folgenden Text:
{}

Antworte in folgendem Format:
<Kategorie>: <extrahierter Text>

z.B.
{}

Die Informationen MÜSSEN wörtlich aus dem Text extrahiert werden, sie dürfen nicht generiert werden!
"""

    english_system_prompt = """
You are a system to support the analysis of large amounts of text. You will assist the user by extracting the required information from the provided document. You will always answer in the required format and use no other formatting than expected by the user!
"""

    english_prompt = """
Here is a list of entity categories that I would like to extract from the text. The categories are:
{}

Please extract fitting entities from the following text:
{}

Respond in the following format:
<category>: <extracted text>

e.g.
{}

Remember, you MUST extract the information verbatim from the text, do not generate it!
"""

    def __init__(self, tokens, ner_tags, id2label, examples, model: ModelsEnum, port: int, lang: str, dataset_name: str, task_name: str, output_dir_path: Path, report_path: Path, prompt=None):
        # call parent
        super(SpanClassificationEvaluator, self).__init__(model=model, port=port, lang=lang, dataset_name=dataset_name, task_name=task_name, output_dir_path=output_dir_path, report_path=report_path)

        assert len(tokens) == len(ner_tags), "The number of tokens and NER tags must be equal."

        self.tokens = tokens
        self.ner_tags = ner_tags

        self.id2label = id2label
        self.label2id = {v.lower(): k for k, v in id2label.items()}

        self.label_text = "\n".join([f"{(idx+1)}. {label.title()}" for idx, label in enumerate(list(id2label.values())[1:])])
        print("Using the following labels:")
        print(self.label_text)
        print()

        self.examples = examples
        print("Using the following examples:")
        print(self.examples.strip())

        if lang == "de":
            self.system_prompt = self.german_system_prompt.strip()
            self.prompt = prompt if prompt is not None else self.german_prompt
        elif lang == "en":
            self.system_prompt = self.english_system_prompt.strip()
            self.prompt = prompt if prompt is not None else self.english_prompt
        else:
            raise ValueError("Language not supported. Please choose 'de' or 'en'.")

    def _parse_response(self, sentence: str, response: str) -> List[int]:
        sentence_tokens = sentence.split()
        sentence_tags = [0] * len(sentence_tokens)

        for line in response.strip().split("\n"):
            if not line.strip():
                continue
            if ":" not in line:
                continue

            splitted_line = line.split(":")
            if len(splitted_line) != 2:
                continue
            
            label = splitted_line[0].strip()
            token = splitted_line[1].strip()
            token_tokens = token.split()

            if len(token_tokens) == 0:
                continue

            if label.startswith("<"):
                label = label[1:]
            if label.endswith(">"):
                label = label[:-1]

            if label.startswith("**"):
                label = label[2:]
            if label.endswith("**"):
                label = label[:-2]

            if label.lower() not in self.label2id:
                continue

            # find all token_tokens in the sentence
            for idx, sentence_token in enumerate(sentence_tokens):
                if sentence_token == token_tokens[0] and sentence_tokens[idx:idx+len(token_tokens)] == token_tokens:
                    sentence_tags = sentence_tags[:idx] + (len(token_tokens) * [self.label2id[label.lower()]]) + sentence_tags[idx+len(token_tokens):]

        return sentence_tags

    def _prompt_ollama(self, sentence: str) -> Tuple[str, List[int]]:
        response = self.client.chat(model=self.model, messages=[
            {
                'role': 'system',
                'content': self.system_prompt,
            },
            {
                'role': 'user',
                'content': self.prompt.format(self.label_text.strip(), sentence, self.examples.strip()).strip(),
            },
        ])
        
        message = response["message"]["content"]
        return message, self._parse_response(sentence, message)

    def _evaluate(self):
        preds = []
        golds = []
        messages = []

        for tokens, ner_tags in tqdm(zip(self.tokens, self.ner_tags), desc="Evaluating"):
            sentence = " ".join(tokens).strip()
            message, predicted_labels = self._prompt_ollama(sentence)

            preds.append([self.id2label[label] for label in predicted_labels])
            golds.append([self.id2label[label] for label in ner_tags])
            messages.append(message)

        # store the evaluation results
        pd.DataFrame({
            "tokens": self.tokens,
            "ner_tags": golds,
            "predicted_tags": preds,
            "message": messages
        }).to_parquet(self.output_file_path.with_suffix(".parquet"))

    def _report(self):
        # load the results
        df = pd.read_parquet(self.output_file_path.with_suffix(".parquet"))
        print("Num Results:", df.size)

        assert df.size > 0, "The evaluation results are empty."

        golds = [x.tolist() for x in df["ner_tags"].to_list()]
        preds = [x.tolist() for x in df["predicted_tags"].to_list()]

        # compute metric
        f1 = f1_score(golds, preds)
        acc = accuracy_score(golds, preds)
        report = classification_report(golds, preds, output_dict=True)
        df_report = pd.DataFrame(report)

        # Print report
        print("F1-Score: ", f1)
        print("Accuracy: ", acc)
        print(classification_report(golds, preds))

        # extract most important information
        precision, recall, f1, support = [round(x * 100.0, 2) for x in df_report["weighted avg"].tolist()]

        # write reports
        df_report.transpose().to_csv(self.output_file_path.with_suffix(".report.csv"))
        self._add_results_to_report({
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Accuracy": round(acc * 100.0, 2),
        })

@app.command()
def fewnerd_coarse(model: ModelsEnum, port: int, report_only: bool = False):
    # load dataset
    dataset = load_dataset("DFKI-SLT/few-nerd", "supervised")
    df = dataset["test"].to_pandas()
    eval_df = df.sample(n=min(len(df), 10000), random_state=42)

    tokens = eval_df["tokens"].tolist()
    ner_tags = eval_df["ner_tags"].tolist()

    # labels
    id2label = {
        0: "O",
        1: "art",
        2: "building",
        3: "event",
        4: "location",
        5: "organization",
        6: "other",
        7: "person",
        8: "product",
    }

    # examples
    examples = """
Art: Mona Lisa
Building: Eiffel Tower
"""

    # start evaluator
    evaluator = SpanClassificationEvaluator(
        model=model,
        port=port,
        lang="en",
        task_name="span-classification-coarse",
        dataset_name="fewnerd",
        output_dir_path=Path("experiments/span-classification/results/"),
        report_path=Path("experiments/span-classification/report.csv"),
        tokens=tokens,
        ner_tags=ner_tags,
        id2label=id2label,
        examples=examples,
    )
    evaluator.start(report_only=report_only)

@app.command()
def fewnerd_fine(model: ModelsEnum, port: int, report_only: bool = False):
    # load dataset
    dataset = load_dataset("DFKI-SLT/few-nerd", "supervised")
    df = dataset["test"].to_pandas()
    eval_df = df.sample(n=min(len(df), 10000), random_state=42)

    tokens = eval_df["tokens"].tolist()
    ner_tags = eval_df["fine_ner_tags"].tolist()

    # labels
    id2label = {
        0: "O",
        1: "art - broadcastprogram",
        2: "art - film",
        3: "art - music",
        4: "art - other",
        5: "art - painting",
        6: "art - writtenart",
        7: "building - airport",
        8: "building - hospital",
        9: "building - hotel",
        10: "building - library",
        11: "building - other",
        12: "building - restaurant",
        13: "building - sportsfacility",
        14: "building - theater",
        15: "event - attack/battle/war/militaryconflict",
        16: "event - disaster",
        17: "event - election",
        18: "event - other",
        19: "event - protest",
        20: "event - sportsevent",
        21: "location - GPE",
        22: "location - bodiesofwater",
        23: "location - island",
        24: "location - mountain",
        25: "location - other",
        26: "location - park",
        27: "location - road/railway/highway/transit",
        28: "organization - company",
        29: "organization - education",
        30: "organization - government/governmentagency",
        31: "organization - media/newspaper",
        32: "organization - other",
        33: "organization - politicalparty",
        34: "organization - religion",
        35: "organization - showorganization",
        36: "organization - sportsleague",
        37: "organization - sportsteam",
        38: "other - astronomything",
        39: "other - award",
        40: "other - biologything",
        41: "other - chemicalthing",
        42: "other - currency",
        43: "other - disease",
        44: "other - educationaldegree",
        45: "other - god",
        46: "other - language",
        47: "other - law",
        48: "other - livingthing",
        49: "other - medical",
        50: "person - actor",
        51: "person - artist/author",
        52: "person - athlete",
        53: "person - director",
        54: "person - other",
        55: "person - politician",
        56: "person - scholar",
        57: "person - soldier",
        58: "product - airplane",
        59: "product - car",
        60: "product - food",
        61: "product - game",
        62: "product - other",
        63: "product - ship",
        64: "product - software",
        65: "product - train",
        66: "product - weapon"
    }

    # examples
    examples = """
Art - Painting: Mona Lisa
Building - Other: Eiffel Tower
"""

    # start evaluator
    evaluator = SpanClassificationEvaluator(
        model=model,
        port=port,
        lang="en",
        task_name="span-classification-fine",
        dataset_name="fewnerd",
        output_dir_path=Path("experiments/span-classification/results/"),
        report_path=Path("experiments/span-classification/report.csv"),
        tokens=tokens,
        ner_tags=ner_tags,
        id2label=id2label,
        examples=examples,
    )
    evaluator.start(report_only=report_only)

@app.command()
def germanler_coarse(model: ModelsEnum, port: int, report_only: bool = False):
    # load dataset
    df = pd.read_parquet("datasets/german-ler/german_ler_test.parquet")
    eval_df = df.sample(n=min(len(df), 10000), random_state=42)

    tokens = eval_df["tokens"].tolist()
    ner_tags = eval_df["ner_tags"].tolist()

    # labels
    id2label = {
        0: 'O',
        1: 'person',
        2: 'ort',
        3: 'organisation',
        4: 'norm',
        5: 'gesetz',
        6: 'rechtsprechung',
        7: 'literatur'
    }

    # examples
    examples = """
Person: Angela Merkel
Gesetz: Artikel 5
"""

    # start evaluator
    evaluator = SpanClassificationEvaluator(
        model=model,
        port=port,
        lang="de",
        task_name="span-classification-coarse",
        dataset_name="germanler",
        output_dir_path=Path("experiments/span-classification/results/"),
        report_path=Path("experiments/span-classification/report.csv"),
        tokens=tokens,
        ner_tags=ner_tags,
        id2label=id2label,
        examples=examples,
    )
    evaluator.start(report_only=report_only)

@app.command()
def germanler_fine(model: ModelsEnum, port: int, report_only: bool = False):
    # load dataset
    df = pd.read_parquet("datasets/german-ler/german_ler_test.parquet")
    eval_df = df.sample(n=min(len(df), 10000), random_state=42)

    tokens = eval_df["tokens"].tolist()
    ner_tags = eval_df["fine_ner_tags"].tolist()

    # labels
    id2label = {
        0: 'O',
        1: 'Person',
        2: 'Anwalt',
        3: 'Richter',
        4: 'Land',
        5: 'Stadt',
        6: 'Straße',
        7: 'Landschaft',
        8: 'Organisation',
        9: 'Unternehmen',
        10: 'Institution',
        11: 'Gericht',
        12: 'Marke',
        13: 'Gesetz',
        14: 'Verordnung',
        15: 'EU Norm',
        16: 'Vorschrift',
        17: 'Vertrag',
        18: 'Gerichtsentscheidung',
        19: 'Literatur'
    }

    examples = """
Person: Angela Merkel
Gesetz: Artikel 5
"""

    # start evaluator
    evaluator = SpanClassificationEvaluator(
        model=model,
        port=port,
        lang="de",
        task_name="span-classification-fine",
        dataset_name="germanler",
        output_dir_path=Path("experiments/span-classification/results/"),
        report_path=Path("experiments/span-classification/report.csv"),
        tokens=tokens,
        ner_tags=ner_tags,
        id2label=id2label,
        examples=examples,
    )
    evaluator.start(report_only=report_only)  


@app.command()
def direct_quotation(model: ModelsEnum, port: int, report_only: bool = False):
    # load dataset
    df = pd.read_parquet("datasets/german-quotations/german_direct_quotations.parquet")
    eval_df = df.sample(n=min(len(df), 10000), random_state=42)

    tokens = eval_df["tokens"].tolist()
    ner_tags = eval_df["tags"].tolist()

    # labels
    id2label = {
        0: 'O',
        1: 'Sprecher',
        2: 'Direkte Rede',
    }

    # examples
    examples = """
Sprecher: Angela Merkel
Direkte Rede: "Wir schaffen das!"
"""

    # override default prompt
    prompt = """
Hier ist eine Liste von Informationen, die ich aus dem Text extrahieren möchte. Die Klassen sind:
{}

Bitte extrahiere alle passenden Passagen (falls es welche gibt) aus dem folgenden Text:
{}

Antworte in folgendem Format:
<Klasse>: <extrahierter Text>

z.B.
{}

Die Informationen MÜSSEN wörtlich aus dem Text extrahiert werden, sie dürfen nicht generiert werden!
"""


    # start evaluator
    evaluator = SpanClassificationEvaluator(
        model=model,
        port=port,
        lang="de",
        task_name="span-classification",
        dataset_name="quotations",
        output_dir_path=Path("experiments/span-classification/results/"),
        report_path=Path("experiments/span-classification/report.csv"),
        tokens=tokens,
        ner_tags=ner_tags,
        id2label=id2label,
        examples=examples,
        prompt=prompt
    )
    evaluator.start(report_only=report_only)  


if __name__ == "__main__":
    app()
