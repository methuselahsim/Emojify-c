#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_LINE 2048
#define MAX_EMOJIS 5
#define MAX_SENTENCES 200
#define MAX_WORDS 400000
#define EMBEDDING_DIM 50
#define HIDDEN_DIM 32
#define MAX_WORD_LEN 50
#define MAX_SENTENCE_LEN 256
#define LEARNING_RATE 0.01
#define EPOCHS 500
#define WEIGHT_DECAY 0.0001 // Added regularization

// Structure for word embeddings
typedef struct {
    char word[MAX_WORD_LEN];
    double vector[EMBEDDING_DIM];
} WordVector;

// Emoji mapping
const char *emoji_map[MAX_EMOJIS][3] = {
    {"0", "\xE2\x9D\xA4\xEF\xB8\x8F", "Heart"}, // ❤️
    {"1", "\xE2\x9A\xBE", "Baseball"}, // ⚾
    {"2", "\xF0\x9F\x98\x83", "Grinning Face"}, // 😃
    {"3", "\xF0\x9F\x98\x9E", "Disappointed Face"}, // 😞
    {"4", "\xF0\x9F\x8D\xB4", "Fork and Knife"} // 🍴
};

// Trim whitespace, commas, and quotes
void trim(char *str) {
    char *end = str + strlen(str) - 1;
    while (end > str && (*end == ' ' || *end == '\n' || *end == '\r' || *end == ',' || *end == '\'' || *end == '"')) {
        *end = '\0';
        end--;
    }
    while (*str == ' ' || *str == ',' || *str == '\'' || *str == '"') str++;
    memmove(str, str, strlen(str) + 1);
}

// Parse quoted sentences
int parse_sentences(const char *line, char sentences[][MAX_SENTENCE_LEN], int max_sentences) {
    int count = 0;
    char *copy = strdup(line);
    if (!copy) {
        printf("Error: Memory allocation failed in parse_sentences\n");
        return 0;
    }

    char *ptr = copy;
    int in_quote = 0;
    char *start = NULL;

    printf("Parsing line: %.50s...\n", line);
    while (*ptr && count < max_sentences) {
        if (*ptr == '\'') {
            if (in_quote) {
                *ptr = '\0';
                if (start && strlen(start) > 0) {
                    strncpy(sentences[count], start, MAX_SENTENCE_LEN - 1);
                    sentences[count][MAX_SENTENCE_LEN - 1] = '\0';
                    trim(sentences[count]);
                    if (strlen(sentences[count]) > 0) {
                        printf("Parsed sentence %d: %s\n", count + 1, sentences[count]);
                        count++;
                    }
                }
                in_quote = 0;
            } else {
                in_quote = 1;
                start = ptr + 1;
            }
        }
        ptr++;
    }
    free(copy);
    return count;
}

// Find emoji by label
int find_emoji(int label, const char **emoji, const char **description) {
    if (label < 0 || label >= MAX_EMOJIS) {
        *emoji = "Unknown";
        *description = "Invalid Label";
        return 0;
    }
    *emoji = emoji_map[label][1];
    *description = emoji_map[label][2];
    return 1;
}

// Read GloVe vectors
WordVector *word_vectors = NULL;
int word_vector_count = 0;

int read_glove(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: Could not open GloVe file %s\n", filename);
        return 0;
    }

    word_vectors = malloc(MAX_WORDS * sizeof(WordVector));
    if (!word_vectors) {
        printf("Error: Memory allocation failed for word_vectors\n");
        fclose(file);
        return 0;
    }

    char line[MAX_LINE];
    word_vector_count = 0;
    int line_number = 0;

    printf("Reading GloVe file %s...\n", filename);
    while (fgets(line, MAX_LINE, file) && word_vector_count < MAX_WORDS) {
        line_number++;
        char *token = strtok(line, " ");
        if (!token) {
            printf("Warning: Empty line at %d\n", line_number);
            continue;
        }
        strncpy(word_vectors[word_vector_count].word, token, MAX_WORD_LEN - 1);
        word_vectors[word_vector_count].word[MAX_WORD_LEN - 1] = '\0';

        int i;
        for (i = 0; i < EMBEDDING_DIM; i++) {
            token = strtok(NULL, " ");
            if (!token) {
                printf("Warning: Incomplete vector for word '%s' at line %d\n", 
                       word_vectors[word_vector_count].word, line_number);
                break;
            }
            word_vectors[word_vector_count].vector[i] = atof(token);
        }
        if (i == EMBEDDING_DIM) { // Only increment if full vector read
            word_vector_count++;
        }
        if (word_vector_count % 10000 == 0) {
            printf("Read %d words so far...\n", word_vector_count);
        }
    }
    fclose(file);
    printf("Read %d words from GloVe file\n", word_vector_count);
    return 1;
}

// Find word vector
int find_word_vector(const char *word, double *vector) {
    if (!word_vectors) return 0;
    for (int i = 0; i < word_vector_count; i++) {
        if (strcmp(word, word_vectors[i].word) == 0) {
            memcpy(vector, word_vectors[i].vector, EMBEDDING_DIM * sizeof(double));
            return 1;
        }
    }
    memset(vector, 0, EMBEDDING_DIM * sizeof(double));
    return 0;
}

// MLP weights and biases
double w1[EMBEDDING_DIM][HIDDEN_DIM]; // Input to hidden
double w2[HIDDEN_DIM][MAX_EMOJIS];    // Hidden to output
double b1[HIDDEN_DIM];                // Hidden biases
double b2[MAX_EMOJIS];                // Output biases

// Initialize weights
void init_weights() {
    srand(time(NULL));
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            w1[i][j] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
        }
    }
    for (int i = 0; i < HIDDEN_DIM; i++) {
        for (int j = 0; j < MAX_EMOJIS; j++) {
            w2[i][j] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
        }
        b1[i] = 0.0;
    }
    for (int i = 0; i < MAX_EMOJIS; i++) {
        b2[i] = 0.0;
    }
}

// Read weights with emoji labels
int read_weights(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: Could not open weights file %s\n", filename);
        return 0;
    }

    char line[MAX_LINE];
    // Read w1 (input to hidden)
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        if (!fgets(line, MAX_LINE, file)) {
            printf("Error: Failed to read w1 at line %d\n", i);
            fclose(file);
            return 0;
        }
        char *token = strtok(line, " ");
        for (int j = 0; j < HIDDEN_DIM; j++) {
            if (!token) {
                printf("Error: Incomplete w1 at [%d][%d]\n", i, j);
                fclose(file);
                return 0;
            }
            w1[i][j] = atof(token);
            token = strtok(NULL, " ");
        }
    }
    // Read w2 (hidden to output) with emoji labels
    for (int i = 0; i < HIDDEN_DIM; i++) {
        if (!fgets(line, MAX_LINE, file)) {
            printf("Error: Failed to read w2 at line %d\n", i);
            fclose(file);
            return 0;
        }
        char *token = strtok(line, " ");
        for (int j = 0; j < MAX_EMOJIS; j++) {
            if (!token) {
                printf("Error: Incomplete w2 at [%d][%d]\n", i, j);
                fclose(file);
                return 0;
            }
            w2[i][j] = atof(token);
            token = strtok(NULL, " ");
        }
        // Skip emoji label
        while (token && strcmp(token, "\n") != 0) {
            token = strtok(NULL, " ");
        }
    }
    // Read b1 (hidden biases)
    if (!fgets(line, MAX_LINE, file)) {
        printf("Error: Failed to read b1\n");
        fclose(file);
        return 0;
    }
    char *token = strtok(line, " ");
    for (int i = 0; i < HIDDEN_DIM; i++) {
        if (!token) {
            printf("Error: Incomplete b1 at [%d]\n", i);
            fclose(file);
            return 0;
        }
        b1[i] = atof(token);
        token = strtok(NULL, " ");
    }
    // Read b2 (output biases) with emoji labels
    if (!fgets(line, MAX_LINE, file)) {
        printf("Error: Failed to read b2\n");
        fclose(file);
        return 0;
    }
    token = strtok(line, " ");
    for (int i = 0; i < MAX_EMOJIS; i++) {
        if (!token) {
            printf("Error: Incomplete b2 at [%d]\n", i);
            fclose(file);
            return 0;
        }
        b2[i] = atof(token);
        token = strtok(NULL, " ");
    }
    // Skip emoji label
    while (token && strcmp(token, "\n") != 0) {
        token = strtok(NULL, " ");
    }
    fclose(file);
    printf("Successfully read weights from %s\n", filename);
    return 1;
}

// Save weights with emoji labels
void save_weights(const char *filename) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        printf("Error: Could not open weights file %s\n", filename);
        return;
    }
    // Save w1
    fprintf(file, "# Input to hidden weights (%dx%d)\n", EMBEDDING_DIM, HIDDEN_DIM);
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            fprintf(file, "%.6f ", w1[i][j]);
        }
        fprintf(file, "\n");
    }
    // Save w2 with emoji labels
    fprintf(file, "# Hidden to output weights (%dx%d) with emoji labels\n", HIDDEN_DIM, MAX_EMOJIS);
    for (int i = 0; i < HIDDEN_DIM; i++) {
        for (int j = 0; j < MAX_EMOJIS; j++) {
            fprintf(file, "%.6f ", w2[i][j]);
        }
        fprintf(file, "# Hidden %d to emojis: ", i);
        for (int j = 0; j < MAX_EMOJIS; j++) {
            fprintf(file, "%s (%s) ", emoji_map[j][1], emoji_map[j][2]);
        }
        fprintf(file, "\n");
    }
    // Save b1
    fprintf(file, "# Hidden biases (%d)\n", HIDDEN_DIM);
    for (int i = 0; i < HIDDEN_DIM; i++) {
        fprintf(file, "%.6f ", b1[i]);
    }
    fprintf(file, "\n");
    // Save b2 with emoji labels
    fprintf(file, "# Output biases (%d) with emoji labels\n", MAX_EMOJIS);
    for (int i = 0; i < MAX_EMOJIS; i++) {
        fprintf(file, "%.6f ", b2[i]);
    }
    fprintf(file, "# Emojis: ");
    for (int i = 0; i < MAX_EMOJIS; i++) {
        fprintf(file, "%s (%s) ", emoji_map[i][1], emoji_map[i][2]);
    }
    fprintf(file, "\n");
    fclose(file);
    printf("Weights saved to %s with emoji labels\n", filename);
}

// Sigmoid activation
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_deriv(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

// Softmax
void softmax(double *input, double *output, int size) {
    double max = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max) max = input[i];
    }
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        output[i] = exp(input[i] - max);
        sum += output[i];
    }
    for (int i = 0; i < size; i++) {
        output[i] = sum > 0 ? output[i] / sum : 0;
    }
}

// Forward pass
void forward(double *input, double *hidden, double *output) {
    for (int i = 0; i < HIDDEN_DIM; i++) {
        hidden[i] = b1[i];
        for (int j = 0; j < EMBEDDING_DIM; j++) {
            hidden[i] += input[j] * w1[j][i];
        }
        hidden[i] = sigmoid(hidden[i]);
    }
    for (int i = 0; i < MAX_EMOJIS; i++) {
        output[i] = b2[i];
        for (int j = 0; j < HIDDEN_DIM; j++) {
            output[i] += hidden[j] * w2[j][i];
        }
    }
    softmax(output, output, MAX_EMOJIS);
}

// Backward pass with weight decay
void backward(double *input, double *hidden, double *output, int target, double *delta_output, double *delta_hidden) {
    for (int i = 0; i < MAX_EMOJIS; i++) {
        double target_val = (i == target) ? 1.0 : 0.0;
        delta_output[i] = output[i] - target_val;
    }
    for (int i = 0; i < HIDDEN_DIM; i++) {
        for (int j = 0; j < MAX_EMOJIS; j++) {
            w2[i][j] -= LEARNING_RATE * (delta_output[j] * hidden[i] + WEIGHT_DECAY * w2[i][j]);
        }
    }
    for (int i = 0; i < MAX_EMOJIS; i++) {
        b2[i] -= LEARNING_RATE * delta_output[i];
    }
    for (int i = 0; i < HIDDEN_DIM; i++) {
        delta_hidden[i] = 0.0;
        for (int j = 0; j < MAX_EMOJIS; j++) {
            delta_hidden[i] += delta_output[j] * w2[i][j];
        }
        delta_hidden[i] *= sigmoid_deriv(hidden[i]);
    }
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            w1[i][j] -= LEARNING_RATE * (delta_hidden[j] * input[i] + WEIGHT_DECAY * w1[i][j]);
        }
    }
    for (int i = 0; i < HIDDEN_DIM; i++) {
        b1[i] -= LEARNING_RATE * delta_hidden[i];
    }
}

// Train MLP
void train_mlp(double inputs[][EMBEDDING_DIM], int *labels, int n_samples) {
    init_weights();
    double hidden[HIDDEN_DIM], output[MAX_EMOJIS], delta_output[MAX_EMOJIS], delta_hidden[HIDDEN_DIM];

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double loss = 0.0;
        for (int i = 0; i < n_samples; i++) {
            if (labels[i] < 0 || labels[i] >= MAX_EMOJIS) continue;
            forward(inputs[i], hidden, output);
            for (int j = 0; j < MAX_EMOJIS; j++) {
                double target = (j == labels[i]) ? 1.0 : 0.0;
                loss += -target * log(output[j] + 1e-10);
            }
            backward(inputs[i], hidden, output, labels[i], delta_output, delta_hidden);
        }
        loss /= n_samples;
        if (epoch % 50 == 0) { // Reduced logging frequency
            printf("Epoch %d, Loss: %.6f\n", epoch, loss);
        }
    }
}

// Predict emoji
int predict_emoji(const char *sentence, const char **emoji, const char **description) {
    if (!sentence || strlen(sentence) == 0) {
        *emoji = "Unknown";
        *description = "Invalid Sentence";
        return -1;
    }

    printf("Predicting for sentence: %s\n", sentence);
    double input[EMBEDDING_DIM] = {0};
    int word_count = 0;
    char *copy = strdup(sentence);
    if (!copy) {
        *emoji = "Unknown";
        *description = "Memory Allocation Failed";
        return -1;
    }

    char *word = strtok(copy, " ");
    while (word) {
        double vector[EMBEDDING_DIM] = {0};
        if (find_word_vector(word, vector)) {
            for (int i = 0; i < EMBEDDING_DIM; i++) {
                input[i] += vector[i];
            }
            word_count++;
        }
        word = strtok(NULL, " ");
    }
    free(copy);

    if (word_count == 0) {
        *emoji = "Unknown";
        *description = "No valid words";
        return -1;
    }

    for (int i = 0; i < EMBEDDING_DIM; i++) {
        input[i] /= word_count;
    }

    double hidden[HIDDEN_DIM], output[MAX_EMOJIS];
    forward(input, hidden, output);

    int max_idx = 0;
    for (int i = 1; i < MAX_EMOJIS; i++) {
        if (output[i] > output[max_idx]) {
            max_idx = i;
        }
    }

    *emoji = emoji_map[max_idx][1];
    *description = emoji_map[max_idx][2];
    return max_idx;
}

// Compute confusion matrix
void compute_confusion_matrix(int *actual, int *predicted, int size, int matrix[5][5]) {
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            matrix[i][j] = 0;
        }
    }
    for (int i = 0; i < size; i++) {
        if (actual[i] >= 0 && actual[i] < 5 && predicted[i] >= 0 && predicted[i] < 5) {
            matrix[actual[i]][predicted[i]]++;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 5 || argc > 6) {
        printf("Usage: %s <label_csv_file> <sentence_csv_file> <glove_file> [--train | --load <weights_file>] <output_txt_file>\n", argv[0]);
        return 1;
    }

    char *label_file = NULL, *sentence_file = NULL, *glove_file = NULL, *output_file = NULL, *weights_file = NULL;
    int train_mode = 0;

    // Parse arguments
    int arg_index = 1;
    label_file = argv[arg_index++];
    sentence_file = argv[arg_index++];
    glove_file = argv[arg_index++];

    if (argc == 5) {
        train_mode = 1;
        output_file = argv[arg_index];
    } else {
        if (strcmp(argv[arg_index], "--train") == 0) {
            train_mode = 1;
            output_file = argv[arg_index + 1];
        } else if (strcmp(argv[arg_index], "--load") == 0) {
            train_mode = 0;
            weights_file = argv[arg_index + 1];
            output_file = argv[arg_index + 2];
        } else {
            printf("Error: Invalid option %s. Use --train or --load <weights_file>\n", argv[arg_index]);
            return 1;
        }
    }

    // Read GloVe vectors
    printf("Starting GloVe read...\n");
    if (!read_glove(glove_file)) {
        return 1;
    }

    // Read sentences
    FILE *sentence_file_ptr = fopen(sentence_file, "r");
    if (!sentence_file_ptr) {
        printf("Error: Could not open sentence file %s\n", sentence_file);
        free(word_vectors);
        return 1;
    }

    char sentences[MAX_SENTENCES][MAX_SENTENCE_LEN];
    int sentence_count = 0;
    char line[MAX_LINE];

    printf("Reading sentence file %s...\n", sentence_file);
    while (fgets(line, MAX_LINE, sentence_file_ptr) && sentence_count < MAX_SENTENCES) {
        int parsed = parse_sentences(line, sentences + sentence_count, MAX_SENTENCES - sentence_count);
        sentence_count += parsed;
    }
    fclose(sentence_file_ptr);
    printf("Read %d sentences\n", sentence_count);

    // Read labels
    FILE *label_file_ptr = fopen(label_file, "r");
    if (!label_file_ptr) {
        printf("Error: Could not open label file %s\n", label_file);
        free(word_vectors);
        return 1;
    }

    int labels[MAX_SENTENCES];
    int label_count = 0;

    printf("Reading label file %s...\n", label_file);
    while (fgets(line, MAX_LINE, label_file_ptr) && label_count < MAX_SENTENCES) {
        trim(line);
        char *token = strtok(line, " ,");
        while (token && label_count < MAX_SENTENCES) {
            labels[label_count] = atoi(token);
            if (labels[label_count] < 0 || labels[label_count] >= MAX_EMOJIS) {
                printf("Warning: Invalid label %s at position %d\n", token, label_count);
                labels[label_count] = -1;
            }
            label_count++;
            token = strtok(NULL, " ,");
        }
    }
    fclose(label_file_ptr);
    printf("Read %d labels\n", label_count);

    // Handle train or load
    if (train_mode) {
        // Prepare training data
        double inputs[MAX_SENTENCES][EMBEDDING_DIM] = {{0}};
        int valid_samples = 0;
        for (int i = 0; i < sentence_count && i < label_count; i++) {
            if (labels[i] < 0) continue;
            char *copy = strdup(sentences[i]);
            if (!copy) continue;
            int word_count = 0;
            char *word = strtok(copy, " ");
            while (word) {
                double vector[EMBEDDING_DIM] = {0};
                if (find_word_vector(word, vector)) {
                    for (int j = 0; j < EMBEDDING_DIM; j++) {
                        inputs[valid_samples][j] += vector[j];
                    }
                    word_count++;
                }
                word = strtok(NULL, " ");
            }
            free(copy);
            if (word_count > 0) {
                for (int j = 0; j < EMBEDDING_DIM; j++) {
                    inputs[valid_samples][j] /= word_count;
                }
                labels[valid_samples] = labels[i];
                valid_samples++;
            }
        }
        printf("Prepared %d valid samples for training\n", valid_samples);

        // Train MLP
        printf("Starting training...\n");
        train_mlp(inputs, labels, valid_samples);
        save_weights("weights.txt");
    } else {
        printf("Loading weights from %s...\n", weights_file);
        if (!read_weights(weights_file)) {
            printf("Error: Failed to load weights, exiting\n");
            free(word_vectors);
            return 1;
        }
    }

    // Process sentences and predict
    FILE *output = fopen(output_file, "w");
    if (!output) {
        printf("Error: Could not open output file %s\n", output_file);
        free(word_vectors);
        return 1;
    }

    int predicted[MAX_SENTENCES];
    int min_count = (sentence_count < label_count) ? sentence_count : label_count;
    printf("Processing %d sentences for prediction...\n", min_count);
    for (int i = 0; i < min_count; i++) {
        if (labels[i] < 0) {
            fprintf(output, "Line %d: Sentence=\"%s\", Label=Invalid, Emoji=Unknown, Description=Invalid Label\n", 
                    i + 1, sentences[i]);
            printf("Line %d: Invalid label for %s\n", i + 1, sentences[i]);
            continue;
        }
        const char *emoji, *description;
        find_emoji(labels[i], &emoji, &description);
        fprintf(output, "Line %d: Sentence=\"%s\", Label=%d, Emoji=%s, Description=%s\n",
                i + 1, sentences[i], labels[i], emoji, description);
        printf("Line %d: %s %s\n", i + 1, sentences[i], emoji);

        predicted[i] = predict_emoji(sentences[i], &emoji, &description);
        fprintf(output, "Predicted: Label=%d, Emoji=%s, Description=%s\n\n",
                predicted[i], emoji, description);
        printf("Predicted: %s\n\n", emoji);
    }

    // Compute and output confusion matrix
    printf("Computing confusion matrix...\n");
    int confusion_matrix[5][5];
    compute_confusion_matrix(labels, predicted, min_count, confusion_matrix);
    fprintf(output, "Confusion Matrix (Actual vs Predicted):\n");
    fprintf(output, "    0  1  2  3  4\n");
    for (int i = 0; i < 5; i++) {
        fprintf(output, "%d: ", i);
        for (int j = 0; j < 5; j++) {
            fprintf(output, "%2d ", confusion_matrix[i][j]);
        }
        fprintf(output, " # %s (%s)\n", emoji_map[i][1], emoji_map[i][2]);
    }

    // Compute accuracy
    double accuracy = 0;
    for (int i = 0; i < 5; i++) {
        accuracy += confusion_matrix[i][i];
    }
    accuracy /= min_count > 0 ? min_count : 1;
    fprintf(output, "Accuracy: %.4f\n", accuracy);
    printf("Accuracy: %.4f\n", accuracy);

    fclose(output);
    free(word_vectors);
    printf("Results written to %s\n", output_file);
    return 0;
}
