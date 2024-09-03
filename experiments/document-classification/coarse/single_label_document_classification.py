from typing import Optional, Tuple
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import typer
from matplotlib import pyplot as plt
import re
import rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

from experiments.evaluator import ModelsEnum, LLMEvaluator  # noqa: E402

app = typer.Typer()

class SingleLabelDocumentClassificationEvaluator(LLMEvaluator):

    german_system_prompt = """
Du bist ein System zur Unterstützung bei der Analyse großer Textmengen. 
In diesem Projekt "{}" geht es um "{}".
"""

    german_prompt = """
Bitte klassifiziere das folgende Dokument in genau einen der folgenden Kategorien: 
{}.

Bitte anworte im folgenden Format. Du musst keine Begründung angeben.
Kategorie: <Kategorie>
Begründung: <Begründung>

Dokument:
{}
"""

    english_system_prompt = """
You are a system to support the analysis of large amounts of text.
This project "{}" is about "{}".
"""

    english_prompt = """
Please classify the following document in one of the following categories: 
{}.

Please answer in this format. You are not required to provide any reason.
Category: <category>
Reason: <reason>

Document:
{}
"""

    def __init__(self, project_name, project_description, articles, labels, label_dict, model: ModelsEnum, port: int, lang: str, dataset_name: str, task_name: str, output_dir_path: Path, report_path: Path):
        # call parent
        super(SingleLabelDocumentClassificationEvaluator, self).__init__(model=model, port=port, lang=lang, dataset_name=dataset_name, task_name=task_name, output_dir_path=output_dir_path, report_path=report_path)

        self.project_name = project_name
        self.project_description = project_description

        assert len(articles) == len(labels), "The number of articles and labels must be the same."
        self.articles = articles
        self.labels = [l.lower() for l in labels]

        self.labels_string = "\n".join([f"{k}: {v}" for k, v in label_dict.items()])
        self.label_dict = {k.lower(): v for k, v in label_dict.items()}
        assert all([label in self.label_dict for label in self.labels]), "All labels must be in the label dictionary."

        if lang == "de":
            self.system_prompt = self.german_system_prompt.format(project_name, project_description).strip()
            self.prompt = self.german_prompt
            self.category_word = "Kategorie"
            self.reason_word = "Begründung"
        elif lang == "en":
            self.system_prompt = self.english_system_prompt.format(project_name, project_description).strip()
            self.prompt = self.english_prompt
            self.category_word = "Category"
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
            categories = components[0].split(":")[1].strip()

        else:
            # reasoning has not been provided
            reason = f"The answer does not contain a line break'. Could not extract the reason."
        
            # extract the answer
            categories = response.split(":")[1].strip()

        return categories, reason

    def _prompt_ollama(self, article: str):
        response = self.client.chat(model=self.model, messages=[
            {
                'role': 'system',
                'content': self.system_prompt,
            },
            {
                'role': 'user',
                'content': self.prompt.format(self.labels_string, article).strip(),
            },
            ])
        
        message: str = response["message"]["content"]
        return message, self._parse_response(message)
    

    def _evaluate(self):
        predictions = []
        reasons = []
        messages = []
        for article, label in tqdm(zip(self.articles, self.labels), desc="Evaluating"):
            message, (category, reason) = self._prompt_ollama(article)

            predictions.append(category)
            reasons.append(reason)
            messages.append(message)

        # store the evaluation results in a csv file
        pd.DataFrame({
            "Article": self.articles,
            "Label": self.labels,
            "Prediction": predictions,
            "Reason": reasons,
            "Message": messages
        }).to_csv(self.output_file_path, index=False)

    def _report(self):
        # read the evaluation results
        df = pd.read_csv(self.output_file_path)
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

        # convert to lowercase
        df["Prediction"] = df["Prediction"].str.lower()
        df["Label"] = df["Label"].str.lower()

        # remove columns where the prediction is not in the label dictionary
        df = df[df["Prediction"].isin(self.label_dict.keys())]
        results_filtered_len2 = len(df)

        # count values not in the label dictionary
        not_in_labels_count = results_filtered_len - results_filtered_len2
        print(f"Filtered2 count: {results_filtered_len2}")
        print(f"Predicted label is not in expected labels count: {not_in_labels_count}, percentage: {(not_in_labels_count / results_filtered_len) * 100:.2f}%")

        predictions = df["Prediction"].tolist()
        labels = df["Label"].tolist()

        # classification report
        label_names = list(self.label_dict.keys())
        label2id = {label: i for i, label in enumerate(self.label_dict.keys())}
        y_true = [label2id[label] for label in labels]
        y_pred = [label2id[pred] for pred in predictions]
        print(classification_report(y_true, y_pred, labels=list(label2id.values()), target_names=label_names))
        report = classification_report(y_true, y_pred, labels=list(label2id.values()), target_names=label_names, output_dict=True)
        df_report = pd.DataFrame(report)

        # extract most important information
        accuracy = round(df_report["accuracy"][0] * 100.0, 2)
        precision, recall, f1, support = [round(x * 100.0, 2) for x in df_report["weighted avg"].tolist()]

        # write reports
        df_report.transpose().to_csv(self.output_file_path.with_suffix(".report.csv"))
        self._add_results_to_report({
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Accuracy": accuracy,
        })

        # confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(label2id.values()))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
        disp.plot()
        plt.savefig(self.output_file_path.with_suffix(".matrix.png"))


@app.command()
def tagesschau(model: ModelsEnum, port: int, report_only: bool = False):
    # load dataset
    df = pd.read_parquet("datasets/tagesschau/tagesschau_cleaned.parquet")
    eval_df = df.sample(n=min(len(df), 10000), random_state=42)
    articles = eval_df["article"].to_list()
    labels = eval_df["main_tag"].to_list()
    label_dict = {
        "Inland": "Nachrichten über Deutschland. Zum Beispiel: Deutschlandtrend, Gesellschaft, Innenpolitik, Mittendrin.",
        "Ausland": "Nachrichten über das Ausland. Zum Beispiel: Afrika, Amerika, Asien, Europa, Ozeanien.",
        "Wirtschaft": "Nachrichten über die Wirtschaft. Zum Beispiel: Börse, Finanzen, Konjunktur, Technologie, Unternehmen, Verbraucher, Weltwirtschaft.",
        "Wissen": "Nachrichten über Wissenschaft. Zum Beispiel: Forschung, Gesundheit, Klima, Technologie.",
    }

    # start evaluator
    evaluator = SingleLabelDocumentClassificationEvaluator(
        model=model,
        port=port,
        lang="de",
        task_name="single-label",
        dataset_name="Tagesschau",
        output_dir_path=Path("experiments/document-classification/single-label"),
        report_path=Path("experiments/document-classification/report.csv"),
        project_name="Tagesschau",
        project_description="Eine Analyse der Themen in Deutschland basierend auf den Nachrichten & Artikeln der Tageschau.",
        articles=articles,
        labels=labels,
        label_dict=label_dict
    )
    evaluator.start(report_only=report_only)

@app.command()
def bbc(model: ModelsEnum, port: int, report_only: bool = False):
    # load dataset
    df = pd.read_parquet("datasets/bbc/bbc_cleaned.parquet")
    eval_df = df.sample(n=min(len(df), 10000), random_state=42)
    articles = eval_df["content"].to_list()
    labels = eval_df["main_tag"].to_list()

    label_dict = {
        "UK": "News about the United Kingdom (UK). For example, england, scotland, wales, ireland, or politics.",
        "World": "News about other parts of the world. For example, africa, asia, australia, europe, latin, us, or middle-east.",
        "Sport": "News about all kinds of sports. For example boxing, cricket, footbal, formula1, rugby, or tennis.",
        "Misc": "Any other news, including for example business, education, elections, entertainment, arts, health, science, or technology.",
    }

    # start evaluator
    evaluator = SingleLabelDocumentClassificationEvaluator(
        model=model,
        port=port,
        lang="en",
        task_name="single-label",
        dataset_name="BBC",
        output_dir_path=Path("experiments/document-classification/single-label"),
        report_path=Path("experiments/document-classification/report.csv"),
        project_name="BBC",
        project_description="An analysis of the topics discussed in the United Kingdoms (UK) based on the news and articles of BBC.",
        articles=articles,
        labels=labels,
        label_dict=label_dict
    )
    evaluator.start(report_only=report_only)

@app.command()
def imdb(model: ModelsEnum, port: int, report_only: bool = False):
    # load dataset
    df = pd.read_parquet("datasets/imdb/imdb_cleaned.parquet")
    eval_df = df.sample(n=min(len(df), 10000), random_state=42)
    articles = eval_df["description"].to_list()
    labels = eval_df["genre"].to_list()

    label_dict = {
        "Action": "The action genre features fast-paced, thrilling, and intense sequences of physical feats, combat, and excitement. The characters of these stories are involved in daring and often dangerous situations, requiring them to rely on their physical prowess, skills, and quick thinking to overcome challenges and adversaries.",
        "Adventure": "The adventure genre features exciting journeys, quests, or expeditions undertaken by characters who often face challenges, obstacles, and risks in pursuit of a goal. Adventures can take place in a wide range of settings, from exotic and fantastical locations to historical or even everyday environments.",
        "Animation": "Animation is a form of visual storytelling that involves creating visual art and motion through the use of various techniques and technologies. In animation, images are manipulated to create the illusion of movement, bringing characters, objects, and environments to life. Animation can encompass a wide range of styles, themes, and intended audiences, making it a diverse and versatile form of storytelling.",
        "Biography": "The biography, or 'biopic', is a genre that portrays the life story of a real person, often a notable individual or historical figure. They aim to provide a depiction of the subject's personal history, achievements, challenges, and impact on society.",
        "Crime": "The crime genre features criminal activities, investigations, law enforcement, crimes, and the pursuit of justice. Crime stories often revolve around the planning, execution, and consequences of criminal acts, as well as the efforts to solve and prevent such acts. They explore various aspects of criminal behavior, motives, and the moral dilemmas faced by both perpetrators and those seeking to uphold the law.",
        "Family": "The family genre features stories specifically created to be suitable for a wide range of age groups within a family. Family-oriented content is designed to be enjoyed by both children and adults, often providing entertainment that is wholesome, relatable, and appropriate for all members of a family to watch or experience together.",
        "Fantasy": "The fantasy genre features imaginative and often magical worlds, characters, and events. It explores realms beyond the boundaries of reality, featuring elements such as magic, mythical creatures, supernatural powers, and fantastical settings. These stories can take place in entirely fictional worlds or blend fantastical elements with real-world settings.",
        "Film-Noir": "The film noir subgenre emerged in the 1940s and 1950s and is characterized by its dark, moody atmosphere, intricate plots, morally ambiguous characters, and distinctive visual style. These stories often depict crime, mystery, and psychological drama with a strong emphasis on shadows, contrasts, and visual storytelling techniques.",
        "History": "The history genre features recounting and analyzing past events, societies, cultures, and historical figures. This genre aims to provide insights into the development of civilizations, the causes and consequences of historical events, and the impact of individuals and ideas on the course of history.",
        "Horror": "The horror genre features stories that aim to elicit fear, suspense, and a sense of dread in its audience. Horror stories often explore themes related to the unknown, the supernatural, and the macabre, and they frequently evoke strong emotional reactions such as anxiety, terror, and unease.",
        "Mystery": "The mystery genre features the investigation and solving of a puzzle, typically a crime or an enigmatic event. Mysteries are known for their suspenseful narratives, intricate plots, and the challenge they present to readers or viewers to piece together clues and solve the central mystery alongside the characters.",
        "Romance": "The romance genre features the theme of romantic relationships and emotional connections between characters. These stories focus on the development of love, desire, and intimacy between protagonists, often exploring the challenges, conflicts, and triumphs that arise in their relationships.",
        "SciFi": "The sci-fi genre, short for science fiction, features imaginative and futuristic concepts that are often rooted in scientific principles, technology, and possibilities. These stories delve into 'what if' questions and can serve as a platform to address contemporary social, political, and ethical issues by projecting them onto future or alternate settings.",
        "Sports": "The sport genre features the world of sports, capturing the excitement, competition, and personal journeys of athletes, coaches, and teams. The stories cover a wide range of sports and activities, each with its own unique characteristics and themes.",
        "Thriller": "The thriller genre features suspense, tension, and excitement. These stories are known for keeping audiences on the edge of their seats and delivering intense emotional experiences by revolving around high-stakes situations, dangerous conflicts, and the constant anticipation of unexpected events.",
        "War": "The war genre features armed conflicts, both historical and fictional, and the experiences of individuals and groups involved in warfare. This genre explores the physical, emotional, and moral challenges faced by soldiers, civilians, and others affected by war.",
    }

    # start evaluator
    evaluator = SingleLabelDocumentClassificationEvaluator(
        model=model,
        port=port,
        lang="en",
        task_name="single-label",
        dataset_name="imdb",
        output_dir_path=Path("experiments/document-classification/single-label"),
        report_path=Path("experiments/document-classification/report.csv"),
        project_name="Movie Genres",
        project_description="An analysis of movie genres based on the movies descriptions of IMDB.",
        articles=articles,
        labels=labels,
        label_dict=label_dict
    )
    evaluator.start(report_only=report_only)

if __name__ == "__main__":
    app()
