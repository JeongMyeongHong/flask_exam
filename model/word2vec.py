import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import font_manager, rc
rc('font', family=font_manager.FontProperties(fname='C:/Windows/Fonts/malgunsl.ttf').get_name())

class Solution:
    def __init__(self, sentences) -> None:
        self.sentences = sentences
        # tensorflow 연산에서 matplot 사용할 때 한글 처리

        # 단어 벡터를 분석해 볼 임의의 문장들
        
        # 문장을 전부 합친 후 공백으로 단어들을 나누고 고유한 단어들로 리스트 생성

        # *******
        # 옵션 설정
        # *******
        # 학습을 반복할 횟수
        self.training_epoch = 300
        # 학습률
        self.learning_rate = 0.1
        # 한번에 학습할 데이터의 크기
        self.batch_size = 20
        # 단어 벡터를 구성할 임베딩 차원의 크기
        self.embedding_size = 2
        # x, y 그래프로 표현하기 쉽게 2개의 값만 출력
        self.num_sampled = 15
        # word2vec 모델을 학습시키기 위한 nce_loss 함수에서 사용하기 위한 샘플링 크기
        # batch_size 보다는 작아야 함
        self.voc_size = 0
        # 총 단어의 갯수


        self.trained_embeddings = None
        self.inputs = None
        self.labels = None
        self.skip_grams = None
        self.word_list = []
    
    def preprocess(self):
        sentences = self.sentences
        word_sequnce = " ".join(sentences).split()

        word_list = " ".join(sentences).split()
        word_list = list(set(word_list))
        self.voc_size = len(word_list)
        # 텐서플로 데이터타입 3가지
        # tuple () , dict {}, list []
        # 단 텐서 요소의 데이터 타입과 혼동 주의 !!
        # 문자열로 분석하는 것 보다, 숫자로 분석하는 것이 훨씬 용이하므로
        # 리스트에서 문자들을 인덱스로 뽑아서 사용하기 위해
        # 이를 표현하기 위한 연관배열과
        # 단어 리스트에서 단어를 참조할 수 있는 인덱스 배열을 만듭니다.

        word_dict = {W: i for i,W in enumerate(word_list)}
        self.wore_list = word_list

        self.skip_grams = []
        for i in range(1, len(word_sequnce) - 1):
            target = word_dict[word_sequnce[i]]
            context = [word_dict[word_sequnce[i - 1]],
                    word_dict[word_sequnce[i + 1]]]
            # (context, target) :
            # 스킵그램을 만든 후, 저장은 단어의 고유 번호(index) 로 한다
            for W in context:
                self.skip_grams.append([target, W])
            # (target, context[0]), (target, context[1]), ...(target, context[n])
    
    @staticmethod
    def random_batch(data, size):
        random_inputs = []
        random_labels = []
        random_index = np.random.choice(range(len(data)), size, replace=False)
        
        for i in random_index:
            random_inputs.append(data[i][0]) # target
            random_labels.append([data[i][1]])
            # context word..위에서 컨텍스트 단어는 list 타입 선언
        return random_inputs, random_labels

        

    # *****
    # 신경망 모델 구성
    # *****
    def create_model(self):
        batch_size = self.batch_size
        self.inputs = tf.placeholder(tf.int32, shape=[batch_size])
        self.labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        # tf.nn.nce_loss 를 사용하려면 출력값을 이렇게 [batch_size, 1] 구성해야함

        embeddings = tf.Variable(tf.random_uniform([self.voc_size, self.embedding_size], -1.0, 1.0))
        return embeddings
        # word2vec 모델의 결과 값인 임베딩 벡터를 저장할 변수
        # 총 단어의 갯수와 임베딩 갯수를 크기로 하는 두개의 차원을 갖습니다.
        # embedding vector 의 차원에서 학습할 입력값에 대한 행들을 뽑아옵니다.
        """
        예) embeddings     inputs     selected
            [[1, 2, 3]   -> [2, 3]  -> [[2, 3, 4]
            [2, 3, 4]                  [3, 4, 5]]
            [3, 4, 5]
            [4, 5, 6]]
        """


    # *****
    # 신경망 모델 학습
    # ****
    def fit(self):
        embeddings = self.create_model()
        inputs = self.inputs
        voc_size = self.voc_size
        labels = self.labels


        selected_embed = tf.nn.embedding_lookup(embeddings, inputs)

        # nce_loss 함수에서 사용할 변수를 정의 함

        nce_weights = tf.Variable(tf.random_uniform([voc_size, self.embedding_size], -1.0, 1.0))
        nce_biases = tf.Variable(tf.zeros([voc_size]))
        loss = tf.reduce_mean(
            tf.nn.nce_loss(nce_weights, nce_biases, labels, selected_embed, self.num_sampled, voc_size)
        )
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            
            for step in range(1, self.training_epoch + 1):
                batch_inputs, batch_labels = self.random_batch(self.skip_grams, self.batch_size)
                _, loss_val = sess.run([train_op, loss],
                                    {inputs: batch_inputs,
                                        labels: batch_labels})
                if step % 10 == 0:
                    print("loss at step ", step, ": ", loss_val)
        
            self.trained_embeddings = embeddings.eval()
        # with 구문 안에서는 sess.run 대신에 간단히 eval() 함수를 사용할 수 있음

    # ******
    # 임베딩된 word2vec 결과 확인
    # ******
    def eval(self):
        for i, label in enumerate(self.word_list):
            x, y = self.trained_embeddings[i]
            plt.scatter(x, y)
            plt.annotate(label, xy=(x, y), xytext=(5, 2),
                        textcoords = 'offset points', ha = 'right', va = 'bottom')
        plt.show()

if __name__=='__main__':
    tf.disable_v2_behavior()
    sentences = ["나 고양이 좋다",
                    "나 강아지 좋다",
                    "나 동물 좋다",
                    "강아지 고양이 동물",
                    "여자친구 고양이 강아지 좋다",
                    "고양이 생선 우유 좋다",
                    "강아지 생선 싫다 우유 좋다",
                    "강아지 고양이 눈 좋다",
                    "나 여자친구 좋다",
                    "여자친구 나 싫다",
                    "여자친구 나 영화 책 음악 좋다",
                    "나 게임 만화 애니 좋다",
                    "고양이 강아지 싫다",
                    "강아지 고양이 좋다"]
    c = Solution(sentences)
    c.preprocess()
    c.fit()
    c.eval()