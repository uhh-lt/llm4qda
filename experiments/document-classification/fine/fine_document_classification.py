from typing import List, Optional, Tuple
import pandas as pd
from ollama import Client
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import typer
from matplotlib import pyplot as plt
import re
import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

from experiments.evaluator import ModelsEnum, LLMEvaluator  # noqa: E402

app = typer.Typer()


class HierarchicalDocumentClassificationEvaluator(LLMEvaluator):
    german_system_prompt = """
Du bist ein System zur Unterstützung bei der Analyse großer Textmengen. 
In diesem Projekt "{}" geht es um "{}".
"""

    german_prompt = """
Bitte klassifiziere das folgende Dokument in das folgende hierarchische Kategoriensystem, bei dem eine Hauptkategorie hat mehrere Unterkategorien hat. Wähle genau eine Hauptkategorie und genau eine davon abhängige Unterkategorie. Die Unterkategorie muss zur Hauptkategorie passen: 
{}.

Bitte anworte im folgenden Format. Du musst keine Begründung angeben.
Klassifizierung: <Hauptkategorie>, <Unterkategorie>
Begründung: <Begründung>

Dokument:
{}
"""

    english_system_prompt = """
You are a system to support the analysis of large amounts of text.
This project "{}" is about "{}".
"""

    english_prompt = """
Please classify the following document into the following hierarchical category system where a main category has several subcategories. Select exactly one main category and exactly one dependent subcategory. The subcategory has to match the main category: 
{}.

Please answer in this format. You are not required to provide any reason.
Classification: <main category>, <sub category>
Reason: <reason>

Document:
{}
"""

    def __init__(
        self,
        project_name,
        project_description,
        labels_string,
        articles,
        labels,
        model: ModelsEnum,
        port: int,
        lang: str,
        dataset_name: str,
        task_name: str,
        output_dir_path: Path,
        report_path: Path,
    ):
        # call parent
        super(HierarchicalDocumentClassificationEvaluator, self).__init__(
            model=model,
            port=port,
            lang=lang,
            dataset_name=dataset_name,
            task_name=task_name,
            output_dir_path=output_dir_path,
            report_path=report_path,
        )

        self.project_name = project_name
        self.project_description = project_description

        assert len(articles) == len(
            labels
        ), "The number of articles and labels must be the same."
        self.articles = articles
        self.labels = [l.lower() for l in labels]
        self.unique_labels = set(self.labels)
        self.labels_string = labels_string
        print(f"Unique labels: {self.unique_labels}")

        if lang == "de":
            self.system_prompt = self.german_system_prompt.format(
                project_name, project_description
            ).strip()
            self.prompt = self.german_prompt
            self.category_word = "Klassifizierung"
            self.reason_word = "Begründung"
        elif lang == "en":
            self.system_prompt = self.english_system_prompt.format(
                project_name, project_description
            ).strip()
            self.prompt = self.english_prompt
            self.category_word = "Classification"
            self.reason_word = "Reason"
        else:
            raise ValueError("Language not supported. Please choose 'de' or 'en'.")

    def _parse_response(self, response: str) -> Tuple[Optional[str], str]:
        # check that the answer starts with "Kategorie:"
        if not response.lower().startswith(f"{self.category_word.lower()}:"):
            return None, f"The answer has to start with '{self.category_word}:'."

        if "\n" in response:
            # reasoning has been provided (probably)
            components = re.split(r"\n+", response)

            # extract the reason
            if not components[1].lower().startswith(f"{self.reason_word.lower()}:"):
                reason = f"The answer does not contain '{self.reason_word}:'. Could not extract the reason."
            else:
                reason = components[1].split(":")[1].strip()

            # extract the answer
            categories = components[0].split(":")[1].strip().split(",")
            categories = [c.strip().lower() for c in categories]
            category = "/".join(categories)

        else:
            # reasoning has not been provided
            reason = f"The answer does not contain a line break'. Could not extract the reason."

            # extract the answer
            categories = response.split(":")[1].strip().split(",")
            categories = [c.strip().lower() for c in categories]
            category = "/".join(categories)

        return category, reason

    def _prompt_ollama(self, article: str):
        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": self.prompt.format(self.labels_string, article).strip(),
                },
            ],
        )

        message = response["message"]["content"]
        category, reason = self._parse_response(message)
        return message, category, reason

    def _evaluate(self):
        predictions = []
        reasons = []
        messages = []
        for article, label in tqdm(zip(self.articles, self.labels), desc="Evaluating"):
            message, category, reason = self._prompt_ollama(article)

            predictions.append(category)
            reasons.append(reason)
            messages.append(message)

        # store the evaluation results in a csv file
        pd.DataFrame(
            {
                "Article": self.articles,
                "Label": self.labels,
                "Prediction": predictions,
                "Reason": reasons,
                "Message": messages,
            }
        ).to_csv(self.output_file_path, index=False)

    def _report(self):
        # read the evaluation results
        df = pd.read_csv(self.output_file_path)
        results_len = len(df)

        assert results_len > 0, "The evaluation results are empty."

        # remove columns where the prediction is None
        df = df[df["Prediction"].notna()]
        results_filtered_len = len(df)

        # count None values
        none_count = results_len - results_filtered_len
        print(f"Total count: {results_len}")
        print(f"Filtered count: {results_filtered_len}")
        print(
            f"None count: {none_count}, None percentage: {(none_count / results_len) * 100:.2f}%"
        )

        # convert to lowercase
        df["Prediction"] = df["Prediction"].str.lower()
        df["Label"] = df["Label"].str.lower()

        # remove columns where the prediction is not a valid label
        df = df[df["Prediction"].isin(self.unique_labels)]
        results_filtered_len2 = len(df)

        # count values not in the label dictionary
        not_in_labels_count = results_filtered_len - results_filtered_len2
        print(f"Filtered2 count: {results_filtered_len2}")
        print(
            f"Predicted label is not in expected labels count: {not_in_labels_count}, percentage: {(not_in_labels_count / results_filtered_len) * 100:.2f}%"
        )

        predictions = df["Prediction"].tolist()
        labels = df["Label"].tolist()

        # vectorize output
        label_names = list(self.unique_labels)
        label2id = {label: i for i, label in enumerate(label_names)}
        y_true = [label2id[label] for label in labels]
        y_pred = [label2id[pred] for pred in predictions]

        # classification report
        print(
            classification_report(
                y_true, y_pred, labels=list(label2id.values()), target_names=label_names
            )
        )
        report = classification_report(
            y_true,
            y_pred,
            labels=list(label2id.values()),
            target_names=label_names,
            output_dict=True,
        )
        df_report = pd.DataFrame(report)

        # extract most important information
        accuracy = round(df_report["accuracy"][0] * 100.0, 2)
        precision, recall, f1, support = [
            round(x, 2) * 100.0 for x in df_report["weighted avg"].tolist()
        ]

        # write reports
        df_report.transpose().to_csv(self.output_file_path.with_suffix(".report.csv"))
        self._add_results_to_report(
            {
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "Accuracy": accuracy,
            }
        )

        # confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(label2id.values()))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
        disp.plot()
        plt.savefig(self.output_file_path.with_suffix(".png"))


@app.command()
def tagesschau(model: ModelsEnum, port: int, report_only: bool = False):
    # load dataset
    df = pd.read_parquet("datasets/tagesschau/tagesschau_cleaned.parquet")
    eval_df = df.sample(n=min(len(df), 10000), random_state=42)
    articles = eval_df["article"].to_list()
    labels = eval_df["tag"].to_list()

    labels_string = """
Inland: Nachrichten über Deutschland. 
    - Gesellschaft: Nachrichten über die Gesellschaft.
    - Innenpolitik: Nachrichten über die Innenpolitik.
    - Mittendrin: Nachrichten über das Leben in Deutschland.
Ausland: Nachrichten über das Ausland. 
    - Afrika: Nachrichten über Afrika.
    - Amerika: Nachrichten über Amerika.
    - Asien: Nachrichten über Asien.
    - Europa: Nachrichten über Europa.
    - Ozeanien: Nachrichten über Ozeanien.
Wirtschaft: Nachrichten über die Wirtschaft.
    - Börse: Nachrichten über die Börse.
    - Finanzen: Nachrichten über die Finanzen.
    - Konjunktur: Nachrichten über die Konjunktur.
    - Technologie: Nachrichten über die Technologie.
    - Unternehmen: Nachrichten über Unternehmen.
    - Verbraucher: Nachrichten über Verbraucher.
    - Weltwirtschaft: Nachrichten über die Weltwirtschaft.
Wissen: Nachrichten über Wissenschaft. 
    - Forschung: Nachrichten über Forschung.
    - Gesundheit: Nachrichten über Gesundheit.
    - Klima: Nachrichten über das Klima.
    - Technologie: Nachrichten über Technologie.
"""

    # start evaluator
    evaluator = HierarchicalDocumentClassificationEvaluator(
        model=model,
        port=port,
        lang="de",
        task_name="hierarchical",
        dataset_name="Tagesschau",
        output_dir_path=Path("experiments/document-classification/hierarchical"),
        report_path=Path("experiments/document-classification/report.csv"),
        project_name="Tagesschau",
        project_description="Eine Analyse der Themen in Deutschland basierend auf den Nachrichten & Artikeln der Tageschau.",
        articles=articles,
        labels=labels,
        labels_string=labels_string,
    )
    evaluator.start(report_only=report_only)


@app.command()
def bbc(model: ModelsEnum, port: int, report_only: bool = False):
    # load dataset
    df = pd.read_parquet("datasets/bbc/bbc_cleaned.parquet")
    eval_df = df.sample(n=min(len(df), 10000), random_state=42)
    articles = eval_df["content"].to_list()
    labels = eval_df["tag"].to_list()

    labels_string = """
UK: News about the United Kingdom (UK).
    - England: News about England.
    - Scotland: News about Scotland.
    - Wales: News about Wales.
    - Northern-Ireland: News about Northern-Ireland.
    - Politics: News about politics.
World: News about other parts of the world.
    - Africa: News about Africa.
    - Asia: News about Asia.
    - Australia: News about Australia.
    - Europe: News about Europe.
    - Latin-America: News about Latin America.
    - Middle-East: News about the Middle East.
    - US: News about the United States.
Sport: News about all kinds of sports.
    - Athletics: News about athletics.
    - Boxing: News about boxing.
    - Cricket: News about cricket.
    - Football: News about football.
    - Formula1: News about Formula 1.
    - Rugby: News about rugby.
    - Tennis: News about tennis.
Misc: Any other news.
    - Business: News about business.
    - Education: News about education.
    - Election: News about elections.
    - Entertainment: News about entertainment.
    - Health: News about health.
    - Science: News about science.
    - Technology: News about technology.
"""

    # start evaluator
    evaluator = HierarchicalDocumentClassificationEvaluator(
        model=model,
        port=port,
        lang="en",
        task_name="hierarchical",
        dataset_name="BBC",
        output_dir_path=Path("experiments/document-classification/hierarchical"),
        report_path=Path("experiments/document-classification/report.csv"),
        project_name="BBC",
        project_description="An analysis of the topics discussed in the United Kingdoms (UK) based on the news and articles of BBC.",
        articles=articles,
        labels=labels,
        labels_string=labels_string,
    )
    evaluator.start(report_only=report_only)


if __name__ == "__main__":
    app()
