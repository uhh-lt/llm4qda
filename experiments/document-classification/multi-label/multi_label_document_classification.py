from typing import Optional, Tuple
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, multilabel_confusion_matrix, accuracy_score
import typer
from matplotlib import pyplot as plt
import re
import rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

from experiments.evaluator import ModelsEnum, LLMEvaluator  # noqa: E402

app = typer.Typer()

class MultiLabelDocumentClassificationEvaluator(LLMEvaluator):

    german_system_prompt = """
Du bist ein System zur Unterstützung bei der Analyse großer Textmengen. 
In diesem Projekt "{}" geht es um "{}".
"""

    german_prompt = """
Bitte klassifiziere das folgende Dokument in alle passenden folgenden Kategorien. Es sind mehrere Kategorien möglich: 
{}.

Bitte anworte im folgenden Format. Du musst keine Begründung angeben.
Kategorie: <Kategorie 1>, <Kategorie 2>, ...
Begründung: <Begründung>

Dokument:
{}
"""

    english_system_prompt = """
You are a system to support the analysis of large amounts of text.
This project "{}" is about "{}".
"""

    english_prompt = """
Please classify the following document into all appropriate categories below. Multiple categories are possible:
{}.

Please answer in this format. You are not required to provide any reason.
Category: <category 1>, <category 2>, ...
Reason: <reason>

Document:
{}
"""

    def __init__(self, project_name, project_description, label_dict, articles, labels, model: ModelsEnum, port: int, lang: str, dataset_name: str, task_name: str, output_dir_path: Path, report_path: Path):
        # call parent
        super(MultiLabelDocumentClassificationEvaluator, self).__init__(model=model, port=port, lang=lang, dataset_name=dataset_name, task_name=task_name, output_dir_path=output_dir_path, report_path=report_path)

        self.project_name = project_name
        self.project_description = project_description

        assert len(articles) == len(labels), "The number of articles and labels must be the same."
        self.articles = articles
        self.labels = [l.lower() for l in labels]
        unique_labels = list(set(label for labels in self.labels for label in labels.split(", ")))

        self.labels_string = "\n".join([f"{k}: {v}" for k, v in label_dict.items()])
        self.label_dict = {k.lower(): v for k, v in label_dict.items()}
        assert all([label in self.label_dict for label in unique_labels]), "All labels must be in the label dictionary."

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

        # prepare the results: lowercase, convert to array
        y_true = df["Label"].str.lower().apply(lambda x: [x.strip() for x in x.split(",")]).tolist()
        y_pred = df["Prediction"].str.lower().apply(lambda x: [x.strip() for x in x.split(",")]).tolist()

        label_names = list(self.label_dict.keys())
        id2label = {i: label for i, label in enumerate(label_names)}

        # convert predictions to vetorized form
        mlb = MultiLabelBinarizer()
        mlb.fit([label_names])
        y_true_vector = mlb.transform(y_true)
        y_pred_vector = mlb.transform(y_pred)

        # classification report
        print(classification_report(y_true_vector, y_pred_vector, target_names=label_names))
        report = classification_report(y_true_vector, y_pred_vector, target_names=label_names, output_dict=True)
        df_report = pd.DataFrame(report)

        # extract most important information
        accuracy = float(round(accuracy_score(y_true_vector, y_pred_vector) * 100.0, 2))
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
        cms = multilabel_confusion_matrix(y_true_vector, y_pred_vector)
        for idx, cm in enumerate(cms):
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.title(id2label[idx])
            plt.savefig(self.output_file_path.with_suffix(f".{id2label[idx]}.png"))
            plt.close()

@app.command()
def imdb(model: ModelsEnum, port: int, report_only: bool = False):
    # load dataset
    df = pd.read_parquet("datasets/imdb/imdb_cleaned.parquet")
    eval_df = df.sample(n=min(len(df), 10000), random_state=42)
    articles = eval_df["description"].to_list()
    labels = eval_df["expanded-genres"].to_list()

    label_dict = {
        "Action": "The action genre features fast-paced, thrilling, and intense sequences of physical feats, combat, and excitement. The characters of these stories are involved in daring and often dangerous situations, requiring them to rely on their physical prowess, skills, and quick thinking to overcome challenges and adversaries.",
        "Adventure": "The adventure genre features exciting journeys, quests, or expeditions undertaken by characters who often face challenges, obstacles, and risks in pursuit of a goal. Adventures can take place in a wide range of settings, from exotic and fantastical locations to historical or even everyday environments.",
        "Animation": "Animation is a form of visual storytelling that involves creating visual art and motion through the use of various techniques and technologies. In animation, images are manipulated to create the illusion of movement, bringing characters, objects, and environments to life. Animation can encompass a wide range of styles, themes, and intended audiences, making it a diverse and versatile form of storytelling.",
        "Biography": "The biography, or 'biopic', is a genre that portrays the life story of a real person, often a notable individual or historical figure. They aim to provide a depiction of the subject's personal history, achievements, challenges, and impact on society.",
        "Comedy": "The comedy genre refers to a category of entertainment that aims to amuse and entertain audiences by using humor, wit, and comedic situations. Comedies are created with the primary intention of eliciting laughter and providing lighthearted enjoyment. They encompass a wide range of styles, tones, and themes, appealing to various tastes and audiences.",
        "Crime": "The crime genre features criminal activities, investigations, law enforcement, crimes, and the pursuit of justice. Crime stories often revolve around the planning, execution, and consequences of criminal acts, as well as the efforts to solve and prevent such acts. They explore various aspects of criminal behavior, motives, and the moral dilemmas faced by both perpetrators and those seeking to uphold the law.",
        "Drama": "The drama genre is a broad category that features stories portraying human experiences, emotions, conflicts, and relationships in a realistic and emotionally impactful way. Dramas delve into the complexities of human life, often exploring themes of love, loss, morality, societal issues, personal growth, with the aim to evoke an emotional response from the audience by presenting relatable and thought-provoking stories.",
        "Family": "The family genre features stories specifically created to be suitable for a wide range of age groups within a family. Family-oriented content is designed to be enjoyed by both children and adults, often providing entertainment that is wholesome, relatable, and appropriate for all members of a family to watch or experience together.",
        "Fantasy": "The fantasy genre features imaginative and often magical worlds, characters, and events. It explores realms beyond the boundaries of reality, featuring elements such as magic, mythical creatures, supernatural powers, and fantastical settings. These stories can take place in entirely fictional worlds or blend fantastical elements with real-world settings.",
        "Film-Noir": "The film noir subgenre emerged in the 1940s and 1950s and is characterized by its dark, moody atmosphere, intricate plots, morally ambiguous characters, and distinctive visual style. These stories often depict crime, mystery, and psychological drama with a strong emphasis on shadows, contrasts, and visual storytelling techniques.",
        "Game-Show": "The game show genre features TV series with contestants competing against each other in various challenges, tasks, or quizzes to win prizes or rewards. Game shows are designed to engage viewers by presenting exciting and often competitive scenarios that involve participants using their knowledge, skills, and strategies to outperform their opponents and win.",
        "History": "The history genre features recounting and analyzing past events, societies, cultures, and historical figures. This genre aims to provide insights into the development of civilizations, the causes and consequences of historical events, and the impact of individuals and ideas on the course of history.",
        "Horror": "The horror genre features stories that aim to elicit fear, suspense, and a sense of dread in its audience. Horror stories often explore themes related to the unknown, the supernatural, and the macabre, and they frequently evoke strong emotional reactions such as anxiety, terror, and unease.",
        "Music": "The music genre features stories showcasing musical performances, music-related documentaries and recordings of live concerts and music festivals.",
        "Musical": "The musical genre features stories that combine music, singing, dancing, acting, and often spoken dialogue to tell a story or convey a narrative. Musicals incorporate various elements of drama, music, and choreography to create a cohesive theatrical experience that engages audiences on multiple levels.",
        "Mystery": "The mystery genre features the investigation and solving of a puzzle, typically a crime or an enigmatic event. Mysteries are known for their suspenseful narratives, intricate plots, and the challenge they present to readers or viewers to piece together clues and solve the central mystery alongside the characters.",
        "News": "The news genre delivers timely and factual information about current events, developments, and issues of public interest. The primary purpose of news content is to inform the audience about what is happening in the world around them, providing them with essential facts, analysis, and context.",
        "Reality-TV": "The reality TV genre, short for 'reality television,' features real-life situations, events, and interactions, often involving ordinary people rather than actors performing scripted roles. Reality TV shows aim to capture unscripted and authentic moments, providing viewers with a glimpse into various aspects of human life, behavior, and experiences.",
        "Romance": "The romance genre features the theme of romantic relationships and emotional connections between characters. These stories focus on the development of love, desire, and intimacy between protagonists, often exploring the challenges, conflicts, and triumphs that arise in their relationships.",
        "Sci-Fi": "The sci-fi genre, short for science fiction, features imaginative and futuristic concepts that are often rooted in scientific principles, technology, and possibilities. These stories delve into 'what if' questions and can serve as a platform to address contemporary social, political, and ethical issues by projecting them onto future or alternate settings.",
        "Sport": "The sport genre features the world of sports, capturing the excitement, competition, and personal journeys of athletes, coaches, and teams. The stories cover a wide range of sports and activities, each with its own unique characteristics and themes.",
        "Talk-Show": "The talk show TV series genre features discussions, conversations, and interviews on various topics, often involving guests who are experts, celebrities, or individuals with unique experiences. These shows can cover a wide range of subjects, including current events, entertainment, politics, lifestyle, human interest stories, and more.",
        "Thriller": "The thriller genre features suspense, tension, and excitement. These stories are known for keeping audiences on the edge of their seats and delivering intense emotional experiences by revolving around high-stakes situations, dangerous conflicts, and the constant anticipation of unexpected events.",
        "War": "The war genre features armed conflicts, both historical and fictional, and the experiences of individuals and groups involved in warfare. This genre explores the physical, emotional, and moral challenges faced by soldiers, civilians, and others affected by war.",
        "Western": "The Western genre features stories set primarily in the 19th-century American Old West and often depict the rugged frontier life, exploring themes of individualism, justice, morality, and the clash between civilization and the untamed wilderness. The genre has its roots in the historical context of westward expansion and the challenges faced by pioneers, settlers, outlaws, and lawmen.",
    }

    evaluator = MultiLabelDocumentClassificationEvaluator(
        model=model,
        port=port,
        lang="en",
        task_name="multi-label",
        dataset_name="imdb",
        output_dir_path=Path("experiments/document-classification/multi-label/imdb"),
        report_path=Path("experiments/document-classification/report.csv"),
        project_name="Movie Genres",
        project_description="An analysis of movie genres based on the movies descriptions of IMDB.",
        articles=articles,
        labels=labels,
        label_dict=label_dict
    )
    evaluator.start(report_only=report_only)

@app.command()
def test():
    print("Hello World!")

if __name__ == "__main__":
    app()
