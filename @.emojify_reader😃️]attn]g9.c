#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_LINE 2048
#define MAX_EMOJIS 5
#define MAX_SENTENCES 200
#define MAX_WORDS 400000
#define EMBEDDING_DIM 5
#define HIDDEN_DIM 3
#define MAX_WORD_LEN 50
#define MAX_SENTENCE_LEN 256
#define MAX_TOKENS 50
#define LEARNING_RATE 0.01
#define EPOCHS 5
#define WEIGHT_DECAY 0.0001
#define NUM_HEADS 2
/*
#define MAX_LINE 2048
#define MAX_EMOJIS 5
#define MAX_SENTENCES 200
#define MAX_WORDS 400000
#define EMBEDDING_DIM 50
#define HIDDEN_DIM 32
#define MAX_WORD_LEN 50
#define MAX_SENTENCE_LEN 256
#define MAX_TOKENS 50
#define LEARNING_RATE 0.01
#define EPOCHS 50
#define WEIGHT_DECAY 0.0001
#define NUM_HEADS 4
*/
typedef struct {
    char word[MAX_WORD_LEN];
    double vector[EMBEDDING_DIM];
} WordVector;

const char *emoji_map[MAX_EMOJIS][3] = {
    {"0", "\xE2\x9D\xA4\xEF\xB8\x8F", "Heart"},
    {"1", "\xE2\x9A\xBE", "Baseball"},
    {"2", "\xF0\x9F\x98\x83", "Grinning Face"},
    {"3", "\xF0\x9F\x98\x9E", "Disappointed Face"},
    {"4", "\xF0\x9F\x8D\xB4", "Fork and Knife"}
};

void trim(char *str) {
    char *end = str + strlen(str) - 1;
    while (end > str && (*end == ' ' || *end == '\n' || *end == '\r' || *end == ',' || *end == '\'' || *end == '"')) {
        *end = '\0';
        end--;
    }
    while (*str == ' ' || *str == ',' || *str == '\'' || *str == '"') str++;
    memmove(str, str, strlen(str) + 1);
}

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
                if (start && strlen(start) > 0 && strlen(start) < MAX_SENTENCE_LEN) {
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
        if (i == EMBEDDING_DIM) {
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

int find_word_vector(const char *word, double *vector) {
    if (!word_vectors || !word) return 0;
    for (int i = 0; i < word_vector_count; i++) {
        if (strcmp(word, word_vectors[i].word) == 0) {
            memcpy(vector, word_vectors[i].vector, EMBEDDING_DIM * sizeof(double));
            return 1;
        }
    }
    memset(vector, 0, EMBEDDING_DIM * sizeof(double));
    return 0;
}

void generate_vocab(const char *filename) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        printf("Error: Could not open vocab file %s\n", filename);
        return;
    }

    fprintf(file, "[PAD]\n[UNK]\n[CLS]\n[SEP]\n");
    for (int i = 0; i < word_vector_count; i++) {
        fprintf(file, "%s\n", word_vectors[i].word);
    }
    for (int i = 0; i < MAX_EMOJIS; i++) {
        fprintf(file, "%s\n", emoji_map[i][1]);
    }
    fclose(file);
    printf("Vocabulary written to %s\n", filename);
}

double Wf[(EMBEDDING_DIM + HIDDEN_DIM) * HIDDEN_DIM];
double Wi[(EMBEDDING_DIM + HIDDEN_DIM) * HIDDEN_DIM];
double Wc[(EMBEDDING_DIM + HIDDEN_DIM) * HIDDEN_DIM];
double Wo[(EMBEDDING_DIM + HIDDEN_DIM) * HIDDEN_DIM];
double bf[HIDDEN_DIM], bi[HIDDEN_DIM], bc[HIDDEN_DIM], bo[HIDDEN_DIM];
double Wy[HIDDEN_DIM * MAX_EMOJIS];
double by[MAX_EMOJIS];
double Wq[HIDDEN_DIM * HIDDEN_DIM];
double Wk[HIDDEN_DIM * HIDDEN_DIM];
double Wv[HIDDEN_DIM * HIDDEN_DIM];
double Wo_multi[HIDDEN_DIM * HIDDEN_DIM];

double *Q_buffer, *K_buffer, *V_buffer, *concat_buffer, *output_buffer, *input_buffer, *scores_buffer, *softmax_buffer;

void init_buffers() {
    Q_buffer = malloc(MAX_TOKENS * HIDDEN_DIM * sizeof(double));
    K_buffer = malloc(MAX_TOKENS * HIDDEN_DIM * sizeof(double));
    V_buffer = malloc(MAX_TOKENS * HIDDEN_DIM * sizeof(double));
    concat_buffer = malloc(MAX_TOKENS * HIDDEN_DIM * sizeof(double));
    output_buffer = malloc(MAX_TOKENS * HIDDEN_DIM * sizeof(double));
    input_buffer = malloc((EMBEDDING_DIM + HIDDEN_DIM) * sizeof(double));
    scores_buffer = malloc(MAX_TOKENS * MAX_TOKENS * sizeof(double));
    softmax_buffer = malloc(MAX_TOKENS * MAX_TOKENS * sizeof(double));
    if (!Q_buffer || !K_buffer || !V_buffer || !concat_buffer || !output_buffer || 
        !input_buffer || !scores_buffer || !softmax_buffer) {
        printf("Error: Failed to allocate buffers\n");
        exit(1);
    }
}

void free_buffers() {
    free(Q_buffer);
    free(K_buffer);
    free(V_buffer);
    free(concat_buffer);
    free(output_buffer);
    free(input_buffer);
    free(scores_buffer);
    free(softmax_buffer);
}

void init_weights() {
    srand(time(NULL));
    for (int i = 0; i < (EMBEDDING_DIM + HIDDEN_DIM) * HIDDEN_DIM; i++) {
        Wf[i] = Wi[i] = Wc[i] = Wo[i] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
    }
    for (int i = 0; i < HIDDEN_DIM; i++) {
        bf[i] = bi[i] = bc[i] = bo[i] = 0.0;
    }
    for (int i = 0; i < HIDDEN_DIM * MAX_EMOJIS; i++) {
        Wy[i] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
    }
    for (int i = 0; i < MAX_EMOJIS; i++) {
        by[i] = 0.0;
    }
    for (int i = 0; i < HIDDEN_DIM * HIDDEN_DIM; i++) {
        Wq[i] = Wk[i] = Wv[i] = Wo_multi[i] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
    }
    init_buffers();
}

int read_weights(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: Could not open model file %s\n", filename);
        return 0;
    }

    char line[MAX_LINE];
    char section[50];
    int rows, cols;

    while (fgets(line, MAX_LINE, file)) {
        trim(line);
        if (line[0] == '#' || strlen(line) == 0) continue;

        if (sscanf(line, "%[^:]: %dx%d", section, &rows, &cols) == 3) {
            double *matrix = NULL;
            double *vector = NULL;
            int is_vector = 0;

            if (strcmp(section, "Wf") == 0) matrix = Wf;
            else if (strcmp(section, "Wi") == 0) matrix = Wi;
            else if (strcmp(section, "Wc") == 0) matrix = Wc;
            else if (strcmp(section, "Wo") == 0) matrix = Wo;
            else if (strcmp(section, "Wy") == 0) matrix = Wy;
            else if (strcmp(section, "Wq") == 0) matrix = Wq;
            else if (strcmp(section, "Wk") == 0) matrix = Wk;
            else if (strcmp(section, "Wv") == 0) matrix = Wv;
            else if (strcmp(section, "Wo_multi") == 0) matrix = Wo_multi;
            else if (strcmp(section, "bf") == 0) { vector = bf; is_vector = 1; }
            else if (strcmp(section, "bi") == 0) { vector = bi; is_vector = 1; }
            else if (strcmp(section, "bc") == 0) { vector = bc; is_vector = 1; }
            else if (strcmp(section, "bo") == 0) { vector = bo; is_vector = 1; }
            else if (strcmp(section, "by") == 0) { vector = by; is_vector = 1; }
            else continue;

            if (is_vector) {
                if (fgets(line, MAX_LINE, file)) {
                    char *token = strtok(line, " ");
                    for (int i = 0; i < cols && token; i++) {
                        vector[i] = atof(token);
                        token = strtok(NULL, " ");
                    }
                }
            } else {
                int index = 0;
                for (int i = 0; i < rows; i++) {
                    if (!fgets(line, MAX_LINE, file)) break;
                    char *token = strtok(line, " ");
                    for (int j = 0; j < cols && token; j++) {
                        matrix[index++] = atof(token);
                        token = strtok(NULL, " ");
                    }
                }
            }
        }
    }
    fclose(file);
    init_buffers();
    printf("Successfully read model from %s\n", filename);
    return 1;
}

void save_weights(const char *filename) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        printf("Error: Could not open model file %s\n", filename);
        return;
    }

    fprintf(file, "# Model weights for emoji prediction LSTM with attention\n");
    fprintf(file, "# Dimensions: EMBEDDING_DIM=%d, HIDDEN_DIM=%d, MAX_EMOJIS=%d, NUM_HEADS=%d\n",
            EMBEDDING_DIM, HIDDEN_DIM, MAX_EMOJIS, NUM_HEADS);

    int index;
    fprintf(file, "Wf: %dx%d\n", EMBEDDING_DIM + HIDDEN_DIM, HIDDEN_DIM);
    index = 0;
    for (int i = 0; i < EMBEDDING_DIM + HIDDEN_DIM; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) fprintf(file, "%.6f ", Wf[index++]);
        fprintf(file, "\n");
    }

    fprintf(file, "Wi: %dx%d\n", EMBEDDING_DIM + HIDDEN_DIM, HIDDEN_DIM);
    index = 0;
    for (int i = 0; i < EMBEDDING_DIM + HIDDEN_DIM; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) fprintf(file, "%.6f ", Wi[index++]);
        fprintf(file, "\n");
    }

    fprintf(file, "Wc: %dx%d\n", EMBEDDING_DIM + HIDDEN_DIM, HIDDEN_DIM);
    index = 0;
    for (int i = 0; i < EMBEDDING_DIM + HIDDEN_DIM; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) fprintf(file, "%.6f ", Wc[index++]);
        fprintf(file, "\n");
    }

    fprintf(file, "Wo: %dx%d\n", EMBEDDING_DIM + HIDDEN_DIM, HIDDEN_DIM);
    index = 0;
    for (int i = 0; i < EMBEDDING_DIM + HIDDEN_DIM; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) fprintf(file, "%.6f ", Wo[index++]);
        fprintf(file, "\n");
    }

    fprintf(file, "bf: %d\n", HIDDEN_DIM);
    for (int i = 0; i < HIDDEN_DIM; i++) fprintf(file, "%.6f ", bf[i]);
    fprintf(file, "\n");

    fprintf(file, "bi: %d\n", HIDDEN_DIM);
    for (int i = 0; i < HIDDEN_DIM; i++) fprintf(file, "%.6f ", bi[i]);
    fprintf(file, "\n");

    fprintf(file, "bc: %d\n", HIDDEN_DIM);
    for (int i = 0; i < HIDDEN_DIM; i++) fprintf(file, "%.6f ", bc[i]);
    fprintf(file, "\n");

    fprintf(file, "bo: %d\n", HIDDEN_DIM);
    for (int i = 0; i < HIDDEN_DIM; i++) fprintf(file, "%.6f ", bo[i]);
    fprintf(file, "\n");

    fprintf(file, "Wy: %dx%d\n", HIDDEN_DIM, MAX_EMOJIS);
    index = 0;
    for (int i = 0; i < HIDDEN_DIM; i++) {
        for (int j = 0; j < MAX_EMOJIS; j++) fprintf(file, "%.6f ", Wy[index++]);
        fprintf(file, "\n");
    }

    fprintf(file, "by: %d\n", MAX_EMOJIS);
    for (int i = 0; i < MAX_EMOJIS; i++) fprintf(file, "%.6f ", by[i]);
    fprintf(file, "\n");

    fprintf(file, "Wq: %dx%d\n", HIDDEN_DIM, HIDDEN_DIM);
    index = 0;
    for (int i = 0; i < HIDDEN_DIM; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) fprintf(file, "%.6f ", Wq[index++]);
        fprintf(file, "\n");
    }

    fprintf(file, "Wk: %dx%d\n", HIDDEN_DIM, HIDDEN_DIM);
    index = 0;
    for (int i = 0; i < HIDDEN_DIM; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) fprintf(file, "%.6f ", Wk[index++]);
        fprintf(file, "\n");
    }

    fprintf(file, "Wv: %dx%d\n", HIDDEN_DIM, HIDDEN_DIM);
    index = 0;
    for (int i = 0; i < HIDDEN_DIM; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) fprintf(file, "%.6f ", Wv[index++]);
        fprintf(file, "\n");
    }

    fprintf(file, "Wo_multi: %dx%d\n", HIDDEN_DIM, HIDDEN_DIM);
    index = 0;
    for (int i = 0; i < HIDDEN_DIM; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) fprintf(file, "%.6f ", Wo_multi[index++]);
        fprintf(file, "\n");
    }

    fprintf(file, "# Emoji map: label, emoji, description\n");
    for (int i = 0; i < MAX_EMOJIS; i++) {
        fprintf(file, "%s %s %s\n", emoji_map[i][0], emoji_map[i][1], emoji_map[i][2]);
    }

    fclose(file);
    printf("Model saved to %s\n", filename);
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_deriv(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

double tanh_act(double x) {
    return tanh(x);
}

double tanh_deriv(double x) {
    double t = tanh(x);
    return 1.0 - t * t;
}

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

void matmul(double *A, double *B, double *result, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            result[i * p + j] = 0;
            for (int k = 0; k < n; k++) {
                result[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }
}

void scaled_dot_product_attention(double *Q, double *K, double *V, int seq_len, int dim, double *output) {
    matmul(Q, K, scores_buffer, seq_len, dim, seq_len);
    for (int i = 0; i < seq_len * seq_len; i++) {
        scores_buffer[i] /= sqrt((double)dim);
    }
    for (int i = 0; i < seq_len; i++) {
        double max_score = scores_buffer[i * seq_len];
        for (int j = 1; j < seq_len; j++) {
            if (scores_buffer[i * seq_len + j] > max_score) max_score = scores_buffer[i * seq_len + j];
        }
        double sum = 0.0;
        for (int j = 0; j < seq_len; j++) {
            softmax_buffer[i * seq_len + j] = exp(scores_buffer[i * seq_len + j] - max_score);
            sum += softmax_buffer[i * seq_len + j];
        }
        for (int j = 0; j < seq_len; j++) {
            softmax_buffer[i * seq_len + j] = sum > 0 ? softmax_buffer[i * seq_len + j] / sum : 0;
        }
    }
    matmul(softmax_buffer, V, output, seq_len, seq_len, dim);
}

void attention(double *h, int seq_len, double *context) {
    int head_dim = HIDDEN_DIM / NUM_HEADS;
    clock_t start = clock();

    memset(context, 0, HIDDEN_DIM * sizeof(double));
    matmul(h, Wq, Q_buffer, seq_len, HIDDEN_DIM, HIDDEN_DIM);
    matmul(h, Wk, K_buffer, seq_len, HIDDEN_DIM, HIDDEN_DIM);
    matmul(h, Wv, V_buffer, seq_len, HIDDEN_DIM, HIDDEN_DIM);

    for (int head = 0; head < NUM_HEADS; head++) {
        double *Q_head = malloc(seq_len * head_dim * sizeof(double));
        double *K_head = malloc(seq_len * head_dim * sizeof(double));
        double *V_head = malloc(seq_len * head_dim * sizeof(double));
        double *output_head = malloc(seq_len * head_dim * sizeof(double));
        if (!Q_head || !K_head || !V_head || !output_head) {
            printf("Error: Failed to allocate head buffers\n");
            free(Q_head); free(K_head); free(V_head); free(output_head);
            exit(1);
        }

        for (int t = 0; t < seq_len; t++) {
            for (int j = 0; j < head_dim; j++) {
                int idx = head * head_dim + j;
                Q_head[t * head_dim + j] = Q_buffer[t * HIDDEN_DIM + idx];
                K_head[t * head_dim + j] = K_buffer[t * HIDDEN_DIM + idx];
                V_head[t * head_dim + j] = V_buffer[t * HIDDEN_DIM + idx];
            }
        }

        scaled_dot_product_attention(Q_head, K_head, V_head, seq_len, head_dim, output_head);

        for (int t = 0; t < seq_len; t++) {
            for (int j = 0; j < head_dim; j++) {
                concat_buffer[t * HIDDEN_DIM + head * head_dim + j] = output_head[t * head_dim + j];
            }
        }

        free(Q_head);
        free(K_head);
        free(V_head);
        free(output_head);
    }

    matmul(concat_buffer, Wo_multi, output_buffer, seq_len, HIDDEN_DIM, HIDDEN_DIM);
    for (int j = 0; j < HIDDEN_DIM; j++) {
        for (int t = 0; t < seq_len; t++) {
            context[j] += output_buffer[t * HIDDEN_DIM + j] / seq_len;
        }
    }

    printf("Attention took %.3f ms\n", ((double)(clock() - start) / CLOCKS_PER_SEC) * 1000);
}

void lstm_step(double *xt, double *h_prev, double *c_prev, double *h, double *c, 
               double *forget_gate, double *input_gate, double *cell_gate, double *output_gate) {
    clock_t start = clock();

    memcpy(input_buffer, xt, EMBEDDING_DIM * sizeof(double));
    memcpy(input_buffer + EMBEDDING_DIM, h_prev, HIDDEN_DIM * sizeof(double));

    matmul(input_buffer, Wf, forget_gate, 1, EMBEDDING_DIM + HIDDEN_DIM, HIDDEN_DIM);
    for (int j = 0; j < HIDDEN_DIM; j++) {
        forget_gate[j] += bf[j];
        forget_gate[j] = sigmoid(forget_gate[j]);
    }

    matmul(input_buffer, Wi, input_gate, 1, EMBEDDING_DIM + HIDDEN_DIM, HIDDEN_DIM);
    for (int j = 0; j < HIDDEN_DIM; j++) {
        input_gate[j] += bi[j];
        input_gate[j] = sigmoid(input_gate[j]);
    }

    matmul(input_buffer, Wc, cell_gate, 1, EMBEDDING_DIM + HIDDEN_DIM, HIDDEN_DIM);
    for (int j = 0; j < HIDDEN_DIM; j++) {
        cell_gate[j] += bc[j];
        cell_gate[j] = tanh_act(cell_gate[j]);
    }

    for (int j = 0; j < HIDDEN_DIM; j++) {
        c[j] = forget_gate[j] * c_prev[j] + input_gate[j] * cell_gate[j];
    }

    matmul(input_buffer, Wo, output_gate, 1, EMBEDDING_DIM + HIDDEN_DIM, HIDDEN_DIM);
    for (int j = 0; j < HIDDEN_DIM; j++) {
        output_gate[j] += bo[j];
        output_gate[j] = sigmoid(output_gate[j]);
        h[j] = output_gate[j] * tanh_act(c[j]);
    }

    printf("LSTM step took %.3f ms\n", ((double)(clock() - start) / CLOCKS_PER_SEC) * 1000);
}

void forward(char *sentence, double *output) {
    if (!sentence || strlen(sentence) >= MAX_SENTENCE_LEN) return;
    clock_t start = clock();
    double *h = malloc(MAX_TOKENS * HIDDEN_DIM * sizeof(double));
    double *c = malloc(MAX_TOKENS * HIDDEN_DIM * sizeof(double));
    if (!h || !c) {
        printf("Error: Failed to allocate h/c buffers\n");
        free(h); free(c);
        return;
    }
    memset(h, 0, MAX_TOKENS * HIDDEN_DIM * sizeof(double));
    memset(c, 0, MAX_TOKENS * HIDDEN_DIM * sizeof(double));
    double forget_gate[HIDDEN_DIM], input_gate[HIDDEN_DIM], cell_gate[HIDDEN_DIM], output_gate[HIDDEN_DIM];
    int seq_len = 0;

    char *copy = strdup(sentence);
    if (!copy) {
        printf("Error: Failed to allocate sentence copy\n");
        free(h); free(c);
        return;
    }

    char *word = strtok(copy, " ");
    while (word && seq_len < MAX_TOKENS) {
        double xt[EMBEDDING_DIM] = {0};
        if (find_word_vector(word, xt)) {
            double *h_prev = seq_len > 0 ? h + (seq_len-1) * HIDDEN_DIM : h;
            double *c_prev = seq_len > 0 ? c + (seq_len-1) * HIDDEN_DIM : c;
            lstm_step(xt, h_prev, c_prev, h + seq_len * HIDDEN_DIM, c + seq_len * HIDDEN_DIM, 
                      forget_gate, input_gate, cell_gate, output_gate);
            seq_len++;
        }
        word = strtok(NULL, " ");
    }
    free(copy);

    double context[HIDDEN_DIM];
    attention(h, seq_len, context);

    matmul(context, Wy, output, 1, HIDDEN_DIM, MAX_EMOJIS);
    for (int i = 0; i < MAX_EMOJIS; i++) {
        output[i] += by[i];
    }
    double soft_output[MAX_EMOJIS];
    softmax(output, soft_output, MAX_EMOJIS);
    memcpy(output, soft_output, MAX_EMOJIS * sizeof(double));

    free(h);
    free(c);
    printf("Forward pass took %.3f ms\n", ((double)(clock() - start) / CLOCKS_PER_SEC) * 1000);
}

void backward(char *sentence, double *output, int target, double *delta_output) {
    if (!sentence || strlen(sentence) >= MAX_SENTENCE_LEN) return;
    clock_t start = clock();
    for (int i = 0; i < MAX_EMOJIS; i++) {
        double target_val = (i == target) ? 1.0 : 0.0;
        delta_output[i] = output[i] - target_val;
    }

    double *h = malloc(MAX_TOKENS * HIDDEN_DIM * sizeof(double));
    double *c = malloc(MAX_TOKENS * HIDDEN_DIM * sizeof(double));
    if (!h || !c) {
        printf("Error: Failed to allocate h/c buffers\n");
        free(h); free(c);
        return;
    }
    memset(h, 0, MAX_TOKENS * HIDDEN_DIM * sizeof(double));
    memset(c, 0, MAX_TOKENS * HIDDEN_DIM * sizeof(double));
    double forget_gate[HIDDEN_DIM], input_gate[HIDDEN_DIM], cell_gate[HIDDEN_DIM], output_gate[HIDDEN_DIM];
    int seq_len = 0;

    char *copy = strdup(sentence);
    if (!copy) {
        printf("Error: Failed to allocate sentence copy\n");
        free(h); free(c);
        return;
    }

    char *word = strtok(copy, " ");
    while (word && seq_len < MAX_TOKENS) {
        double xt[EMBEDDING_DIM] = {0};
        if (find_word_vector(word, xt)) {
            double *h_prev = seq_len > 0 ? h + (seq_len-1) * HIDDEN_DIM : h;
            double *c_prev = seq_len > 0 ? c + (seq_len-1) * HIDDEN_DIM : c;
            lstm_step(xt, h_prev, c_prev, h + seq_len * HIDDEN_DIM, c + seq_len * HIDDEN_DIM, 
                      forget_gate, input_gate, cell_gate, output_gate);
            seq_len++;
        }
        word = strtok(NULL, " ");
    }
    free(copy);

    double context[HIDDEN_DIM];
    attention(h, seq_len, context);

    double delta_h[HIDDEN_DIM] = {0};
    for (int i = 0; i < HIDDEN_DIM; i++) {
        for (int j = 0; j < MAX_EMOJIS; j++) {
            delta_h[i] += delta_output[j] * Wy[i * MAX_EMOJIS + j];
        }
    }

    for (int i = 0; i < HIDDEN_DIM; i++) {
        for (int j = 0; j < MAX_EMOJIS; j++) {
            Wy[i * MAX_EMOJIS + j] -= LEARNING_RATE * (delta_output[j] * context[i] + WEIGHT_DECAY * Wy[i * MAX_EMOJIS + j]);
        }
    }
    for (int i = 0; i < MAX_EMOJIS; i++) {
        by[i] -= LEARNING_RATE * delta_output[i];
    }

    for (int t = seq_len - 1; t >= 0; t--) {
        char *copy2 = strdup(sentence);
        if (!copy2) {
            printf("Error: Failed to allocate sentence copy in backward\n");
            free(h); free(c);
            return;
        }
        char *word = strtok(copy2, " ");
        int idx = 0;
        while (word && idx <= t) {
            if (idx == t) {
                find_word_vector(word, input_buffer);
            }
            word = strtok(NULL, " ");
            idx++;
        }
        free(copy2);
        double *h_prev = (t > 0) ? h + (t-1) * HIDDEN_DIM : h;
        memcpy(input_buffer + EMBEDDING_DIM, h_prev, HIDDEN_DIM * sizeof(double));

        for (int i = 0; i < EMBEDDING_DIM + HIDDEN_DIM; i++) {
            for (int j = 0; j < HIDDEN_DIM; j++) {
                double grad_f = delta_h[j] * sigmoid_deriv(forget_gate[j]) * input_buffer[i];
                double grad_i = delta_h[j] * sigmoid_deriv(input_gate[j]) * input_buffer[i];
                double grad_c = delta_h[j] * tanh_deriv(cell_gate[j]) * input_buffer[i];
                double grad_o = delta_h[j] * sigmoid_deriv(output_gate[j]) * input_buffer[i];
                Wf[i * HIDDEN_DIM + j] -= LEARNING_RATE * (grad_f + WEIGHT_DECAY * Wf[i * HIDDEN_DIM + j]);
                Wi[i * HIDDEN_DIM + j] -= LEARNING_RATE * (grad_i + WEIGHT_DECAY * Wi[i * HIDDEN_DIM + j]);
                Wc[i * HIDDEN_DIM + j] -= LEARNING_RATE * (grad_c + WEIGHT_DECAY * Wc[i * HIDDEN_DIM + j]);
                Wo[i * HIDDEN_DIM + j] -= LEARNING_RATE * (grad_o + WEIGHT_DECAY * Wo[i * HIDDEN_DIM + j]);
            }
        }
        for (int j = 0; j < HIDDEN_DIM; j++) {
            bf[j] -= LEARNING_RATE * (delta_h[j] * sigmoid_deriv(forget_gate[j]));
            bi[j] -= LEARNING_RATE * (delta_h[j] * sigmoid_deriv(input_gate[j]));
            bc[j] -= LEARNING_RATE * (delta_h[j] * tanh_deriv(cell_gate[j]));
            bo[j] -= LEARNING_RATE * (delta_h[j] * sigmoid_deriv(output_gate[j]));
        }
    }

    free(h);
    free(c);
    printf("Backward pass took %.3f ms\n", ((double)(clock() - start) / CLOCKS_PER_SEC) * 1000);
}

void train_lstm(char sentences[][MAX_SENTENCE_LEN], int *labels, int n_samples) {
    init_weights();
    double output[MAX_EMOJIS], delta_output[MAX_EMOJIS];

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        for (int i = 0; i < n_samples; i++) {
            if (labels[i] < 0 || labels[i] >= MAX_EMOJIS) continue;
            forward(sentences[i], output);
            for (int j = 0; j < MAX_EMOJIS; j++) {
                double target = (j == labels[i]) ? 1.0 : 0.0;
                loss += -target * log(output[j] + 1e-10);
            }
            backward(sentences[i], output, labels[i], delta_output);
        }
        loss /= n_samples;
        if (epoch % 10 == 0) {
            printf("Epoch %d, Loss: %.6f, Time: %.3f ms\n", epoch, loss,
                   ((double)(clock() - epoch_start) / CLOCKS_PER_SEC) * 1000);
        }
    }
    free_buffers();
}

int predict_emoji(const char *sentence, const char **emoji, const char **description) {
    if (!sentence || strlen(sentence) == 0 || strlen(sentence) >= MAX_SENTENCE_LEN) {
        *emoji = "Unknown";
        *description = "Invalid Sentence";
        return -1;
    }

    double output[MAX_EMOJIS];
    forward((char *)sentence, output);

    int max_idx = 0;
    for (int i = 1; i < MAX_EMOJIS; i++) {
        if (output[i] > output[max_idx]) max_idx = i;
    }

    *emoji = emoji_map[max_idx][1];
    *description = emoji_map[max_idx][2];
    return max_idx;
}

void compute_confusion_matrix(int *actual, int *predicted, int size, int matrix[5][5]) {
    memset(matrix, 0, 5 * 5 * sizeof(int));
    for (int i = 0; i < size; i++) {
        if (actual[i] >= 0 && actual[i] < 5 && predicted[i] >= 0 && predicted[i] < 5) {
            matrix[actual[i]][predicted[i]]++;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 5 || argc > 6) {
        printf("Usage: %s <label_csv_file> <sentence_csv_file> <glove_file> [--train | --load <model_file>] <output_txt_file>\n", argv[0]);
        return 1;
    }

    char *label_file = argv[1];
    char *sentence_file = argv[2];
    char *glove_file = argv[3];
    char *output_file = NULL;
    char *model_file = NULL;
    int train_mode = 0;

    if (argc == 5) {
        train_mode = 1;
        output_file = argv[4];
    } else {
        if (strcmp(argv[4], "--train") == 0) {
            train_mode = 1;
            output_file = argv[5];
        } else if (strcmp(argv[4], "--load") == 0) {
            train_mode = 0;
            model_file = argv[5];
            output_file = argv[6];
        } else {
            printf("Error: Invalid option %s. Use --train or --load <model_file>\n", argv[4]);
            return 1;
        }
    }

    if (!read_glove(glove_file)) {
        return 1;
    }

    generate_vocab("vocab.txt");

    FILE *sentence_file_ptr = fopen(sentence_file, "r");
    if (!sentence_file_ptr) {
        printf("Error: Could not open sentence file %s\n", sentence_file);
        free(word_vectors);
        return 1;
    }

    char sentences[MAX_SENTENCES][MAX_SENTENCE_LEN];
    int sentence_count = 0;
    char line[MAX_LINE];

    while (fgets(line, MAX_LINE, sentence_file_ptr) && sentence_count < MAX_SENTENCES) {
        int parsed = parse_sentences(line, sentences + sentence_count, MAX_SENTENCES - sentence_count);
        sentence_count += parsed;
    }
    fclose(sentence_file_ptr);

    FILE *label_file_ptr = fopen(label_file, "r");
    if (!label_file_ptr) {
        printf("Error: Could not open label file %s\n", label_file);
        free(word_vectors);
        return 1;
    }

    int labels[MAX_SENTENCES];
    int label_count = 0;

    while (fgets(line, MAX_LINE, label_file_ptr) && label_count < MAX_SENTENCES) {
        trim(line);
        char *token = strtok(line, " ,");
        while (token && label_count < MAX_SENTENCES) {
            labels[label_count] = atoi(token);
            if (labels[label_count] < 0 || labels[label_count] >= MAX_EMOJIS) {
                labels[label_count] = -1;
            }
            label_count++;
            token = strtok(NULL, " ,");
        }
    }
    fclose(label_file_ptr);

    if (train_mode) {
        train_lstm(sentences, labels, sentence_count);
        save_weights("model.txt");
    } else {
        if (!read_weights(model_file)) {
            free(word_vectors);
            return 1;
        }
    }

    FILE *output = fopen(output_file, "w");
    if (!output) {
        printf("Error: Could not open output file %s\n", output_file);
        free(word_vectors);
        return 1;
    }

    fprintf(output, "\xEF\xBB\xBF"); // Write UTF-8 BOM for compatibility
    int predicted[MAX_SENTENCES];
    int min_count = (sentence_count < label_count) ? sentence_count : label_count;
    for (int i = 0; i < min_count; i++) {
        if (labels[i] < 0 || strlen(sentences[i]) == 0) {
            fprintf(output, "Line %d: Sentence=\"%s\", Label=Invalid, Emoji=Unknown, Description=Invalid Label\n", 
                    i + 1, sentences[i]);
            continue;
        }
        const char *emoji, *description;
        find_emoji(labels[i], &emoji, &description);
        fprintf(output, "Line %d: Sentence=\"%s\", Label=%d, Emoji=%s, Description=%s\n",
                i + 1, sentences[i], labels[i], emoji, description);

        predicted[i] = predict_emoji(sentences[i], &emoji, &description);
        fprintf(output, "Predicted: Label=%d, Emoji=%s, Description=%s\n\n",
                predicted[i], emoji, description);
    }

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

    double accuracy = 0;
    for (int i = 0; i < 5; i++) {
        accuracy += confusion_matrix[i][i];
    }
    accuracy /= min_count > 0 ? min_count : 1;
    fprintf(output, "Accuracy: %.4f\n", accuracy);

    fclose(output);
    free(word_vectors);
    return 0;
}
