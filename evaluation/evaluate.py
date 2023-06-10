from sentence_transformers import SentenceTransformer, util
import nltk
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import csv


thresholds = [i / 1000 for i in range(1001)]

completeness_values = []
succinctness_values = []
f_score_values = []

nltk.download("punkt")
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

sources_path = "./evaluation/sources"
summaries_path = "./evaluation/summaries"

sources_files = [f for f in listdir(sources_path) if isfile(join(sources_path, f))]
summaries_files = [
    f for f in listdir(summaries_path) if isfile(join(summaries_path, f))
]


def create_cosine_similarity_matrix(row_tensors, column_tensors):
    num_sources = len(row_tensors)
    num_summaries = len(column_tensors)

    cosine_similarity_matrix = np.zeros((num_sources, num_summaries))

    for i in range(num_sources):
        for j in range(num_summaries):
            if np.array_equal(row_tensors[i], column_tensors[j]):
                cosine_similarity_matrix[i, j] = 0.0
            else:
                cosine_similarity = util.cos_sim(
                    row_tensors[i].unsqueeze(0),
                    column_tensors[j].unsqueeze(0),
                )
                cosine_similarity_matrix[i, j] = cosine_similarity.item()

    return cosine_similarity_matrix


for file_name in sources_files:
    with open("./evaluation/sources/" + file_name, "r", encoding="utf-8") as file:
        text = file.read()

    file_name_without_ext = file_name.replace(".txt", "")

    sources_sentences = nltk.sent_tokenize(text)
    sources_embeddings = model.encode(sources_sentences, convert_to_tensor=True)

    summaries_current_files = list(
        filter(
            lambda item: item.startswith(file_name_without_ext),
            summaries_files,
        )
    )

    for summary_file in summaries_current_files:
        # Initialize lists to store values for plotting
        completeness_values = []
        succinctness_values = []
        f_score_values = []

        with open(
            "./evaluation/summaries/" + summary_file, "r", encoding="utf-8"
        ) as file_name:
            text = file_name.read()

        summaries_sentences = nltk.sent_tokenize(text)
        summaries_embeddings = model.encode(summaries_sentences, convert_to_tensor=True)

        cosine_similarity_matrix_source_summarized = create_cosine_similarity_matrix(
            sources_embeddings, summaries_embeddings
        )
        cosine_similarity_matrix_summarized_summarized = (
            create_cosine_similarity_matrix(summaries_embeddings, summaries_embeddings)
        )

        # Get the maximum value for each row
        source_summarized_max_values = np.max(
            cosine_similarity_matrix_source_summarized, axis=1
        )
        summarized_summarized_max_value = np.max(
            cosine_similarity_matrix_summarized_summarized, axis=1
        )

        results = []

        for threshold in thresholds:
            # for succinctness_threshold in succinctness_thresholds:
            # Count the values that exceed the threshold
            completeness_count = np.sum(source_summarized_max_values >= threshold)
            succinctness_count = np.sum(
                # summarized_summarized_max_value < succinctness_threshold
                summarized_summarized_max_value
                < threshold
            )

            completeness = completeness_count / len(sources_embeddings)
            succinctness = succinctness_count / len(summaries_embeddings)

            f_score = 2 * (completeness * succinctness) / (completeness + succinctness)

            # NOTE: Graphs
            completeness_values.append(completeness)
            succinctness_values.append(succinctness)
            f_score_values.append(f_score)

            # NOTE: CSV
            results.append([threshold, completeness, succinctness, f_score])

        # NOTE: Graphs
        image_path = "./evaluation/results/{}.png".format(summary_file)

        plt.figure(num=summary_file)
        plt.plot(thresholds, completeness_values, label="Completeness")
        plt.plot(thresholds, succinctness_values, label="Succinctness")
        plt.plot(thresholds, f_score_values, label="F-score")
        plt.xlabel("Completeness Threshold")
        plt.ylabel("Value")
        plt.legend()
        plt.title("Graph for {}".format(summary_file))

        plt.savefig(image_path)

        # NOTE: CSV
        csv_file_path = "./evaluation/results/{}.csv".format(summary_file)

        with open(csv_file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ["Threshold", "Completeness", "Succinctness", "F-Score"]
            )  # Write the header
            writer.writerows(results)  # Write the data rows

print("Results exported to", csv_file_path)


# print("Cosine similarity matrix saved to", csv_file_path)
