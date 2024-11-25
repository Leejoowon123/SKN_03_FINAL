import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.model_selection import ParameterGrid
from keras.layers import Lambda, Activation
from model_loading import ModelHandler
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.layers import Dense, Multiply, Permute, Reshape
import matplotlib.pyplot as plt
from DataLoading import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config



class DeepFM_2:
    def __init__(self):
        self.fm_features = config.fm_features
        self.deep_features = config.deep_features
        self.target_name = config.target_column
        self.embedding_dim = 8
        # self.embedding_file = config.embedding_file
        self.model_file_path = config.model_file_path
        self.model_name = config.model_name


    def attention_layer(self, inputs, name="attention"):
        attention = Dense(1, activation='softmax', name=f"{name}_score")(inputs)
        attention = Permute((2, 1), name=f"{name}_permute")(attention)
        output = Multiply(name=f"{name}_multiply")([inputs, attention])
        return output    
    
    def build_model(self, input_dim_dict, optimizer, dropout_rate= 0.3, hidden_units= [256,128, 64], activation='relu', combine_mode='default'):
        inputs = {feature: Input(shape=(1,), name=feature) for feature in self.fm_features + self.deep_features}

        # Embedding layers for FM features
        embeddings = {
            feature: Embedding(input_dim=input_dim_dict[feature], output_dim=self.embedding_dim, name=f"{feature}_embedding")
            for feature in self.fm_features
        }
         # Actor Attention Layer
        if 'cast_id' in input_dim_dict:
            cast_embedding = embeddings['cast_id'](inputs['cast_id'])
            cast_attention = self.attention_layer(cast_embedding, name="actor_attention")
            cast_embedding_flattened = Flatten()(cast_attention)
        else:
            cast_embedding_flattened = None

        # First-order term
        first_order_terms = [Flatten()(embeddings[feature](inputs[feature])) for feature in self.fm_features]
        if cast_embedding_flattened is not None:
            first_order_terms.append(cast_embedding_flattened)

        # Second-order FM interactions
        embeddings_concat = Concatenate(axis=1)([embeddings[feature](inputs[feature]) for feature in self.fm_features])
        sum_square = Lambda(lambda x: tf.square(tf.reduce_sum(x, axis=1)))(embeddings_concat)
        square_sum = Lambda(lambda x: tf.reduce_sum(tf.square(x), axis=1))(embeddings_concat)
        second_order = Lambda(lambda x: 0.5 * (x[0] - x[1]))([sum_square, square_sum])  
        # Deep component
        deep_input = Concatenate(axis=1)([Flatten()(inputs[feature]) for feature in self.deep_features])
        x = deep_input

        # 활성화 함수 객체 확인 및 적용
        activation_function = {
            'relu': Activation('relu'),
            'leakyrelu': tf.keras.layers.LeakyReLU(),
            'elu': tf.keras.layers.ELU(),
            'selu': Activation('selu'),
        }.get(activation, Activation('relu'))

        for units in hidden_units:
            x = Dense(units)(x)
            x = tf.keras.layers.BatchNormalization()(x)  # Batch Normalization 
            x = activation_function(x)
            x = Dropout(dropout_rate)(x)

        # Combine FM and Deep components
        if combine_mode == 'default':
            combined = Concatenate(axis=1)(first_order_terms + [second_order, x])
        elif combine_mode == 'fm_to_deep':
            combined = Concatenate(axis=1)([second_order, x])
        elif combine_mode == 'deep_to_fm':
            first_order_terms = [Flatten()(term) for term in first_order_terms]  # Flatten 처리
            first_order_terms_combined = Concatenate(axis=1)(first_order_terms)  # 결합
            combined = Concatenate(axis=1)([first_order_terms_combined, x])  # 최종 결합
        elif combine_mode == 'parallel':
            # FM 1차 항, 2차 항, Deep 컴포넌트 출력 병렬 결합
            first_order_terms = [Flatten()(term) for term in first_order_terms]  # Flatten 처리
            first_order_terms_combined = Concatenate(axis=1)(first_order_terms)  # FM 1차 항 결합
            
            second_order_combined = Flatten()(second_order)  # FM 2차 항 Flatten

            combined = Concatenate(axis=1)([
                first_order_terms_combined, 
                second_order_combined, 
                x  # Deep 컴포넌트 출력
            ])  # 병렬 결합
        else:
            raise ValueError("Invalid combine_mode")
        
        combined = Concatenate(axis=1)(first_order_terms + [second_order, x])
        output = Dense(1, activation="sigmoid")(combined)

        # Build and compile model
        model = Model(inputs=list(inputs.values()), outputs=output)
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
        return model

    # Prepare data
    def prepare_data(self):
        # data = DataLoader.load_json_to_dataframe(self.embedding_file)
        data = DataLoader.load_json_to_dataframe("C:/SKN_3_MyProject/SKN_03_FINAL/Data/Final/embedding_2.json")
        data = self.ensure_data_types(data)

        # 모델 입력 구조와 데이터 키를 일치시키기
        input_dim_dict = {col: data[col].nunique() + 1 for col in data.columns if col != self.target_name}
        train_data = data.sample(frac=0.8, random_state=42)
        test_data = data.drop(train_data.index)

        x_train = {key: train_data[key].values for key in input_dim_dict.keys()}
        y_train = train_data[self.target_name].values
        x_test = {key: test_data[key].values for key in input_dim_dict.keys()}
        y_test = test_data[self.target_name].values

        # 입력 키와 모델 입력 키가 일치하도록 변환
        x_train = self.adjust_input_structure(x_train)
        x_test = self.adjust_input_structure(x_test)

        return x_train, y_train, x_test, y_test, input_dim_dict
    

    def ensure_data_types(self, df):
        # 주어진 데이터 타입으로 변환

        df["cast_id"], _ = pd.factorize(df["cast_id"])
        df["editor_combined_id"], _ = pd.factorize(df["editor_combined_id"])
        # 빈도수 추가
        df = self.add_id_frequencies(df)
        # df["cast_id"] = df["cast_id"].astype(int)
        df["genre"] = df["genre"].astype(int)
        df["title"] = df["title"].astype(int)
        # df["editor_combined_id"] = df["editor_combined_id"].astype(int)
        df["percentage"] = df["percentage"].astype(float)
        df["musical_license"] = df["musical_license"].astype(int)
        df["period"] = df["period"].astype(float)
        df["ticket_price"] = df["ticket_price"].astype(float)
        df["day_vector"] = df["day_vector"].astype(float)
        df ['time_category'] = df["time_category"].astype(int)
        df[self.target_name] = df[self.target_name].astype(int)
        return df
    
    def add_id_frequencies(self, df):
        # cast_id와 editor_combined_id의 빈도수 추가
        cast_freq = df['cast_id'].value_counts().to_dict()
        df['cast_id_freq'] = df['cast_id'].map(cast_freq)

        editor_freq = df['editor_combined_id'].value_counts().to_dict()
        df['editor_combined_id_freq'] = df['editor_combined_id'].map(editor_freq)

        return df


    def adjust_input_structure(self, x_data):
        # 모델 입력 구조에 맞게 키 재정렬
        expected_keys = config.expected_keys
        adjusted_data = {key: x_data[key] for key in expected_keys if key in x_data}
        return adjusted_data

    def train_and_save_model(self):
        x_train, y_train, x_test, y_test, input_dim_dict = self.prepare_data()

        param_grid = {
            'optimizer': ['adam', 'sgd'], # , 'rmsprop'
            'dropout_rate': [0.3], # , 0.4, 0.5
            'hidden_units': [[128, 64, 32], [256, 128, 64]], # , [64, 32]
            'activation': ['relu'], # , 'leakyrelu', 'elu'
            'combine_mode': ['default', 'fm_to_deep', 'deep_to_fm', 'parallel'],
            'learning_rate': [0.001, 0.01], # , 0.1
            'batch_size': [32, 64], # , 128
        }

        best_model = None
        best_accuracy = 0
        best_params = {}
        total_combinations = len(list(ParameterGrid(param_grid)))
        start_time = time.time()
        print(f"Total parameter combinations: {total_combinations}")

        with tqdm(total=total_combinations, desc="Hyperparameter Search") as pbar:
            for idx, params in enumerate(ParameterGrid(param_grid)):
                elapsed_time = time.time() - start_time
                avg_time_per_iteration = elapsed_time / (idx + 1)
                remaining_iterations = total_combinations - (idx + 1)
                estimated_remaining_time = avg_time_per_iteration * remaining_iterations

                # 남은 시간
                pbar.set_postfix_str(f"ETA: {int(estimated_remaining_time // 60)}m {int(estimated_remaining_time % 60)}s")


                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=7,
                    restore_best_weights=True,
                    verbose=1
                )
                
                print(f"Training with params: {params}")
                
                learning_rate = params.pop('learning_rate')
                optimizer_type = params.pop('optimizer')   
                batch_size = params.pop('batch_size')  
                optimizer = {
                    'adam': Adam(learning_rate=learning_rate),
                    'sgd': SGD(learning_rate=learning_rate),
                    'rmsprop': RMSprop(learning_rate=learning_rate)
                }.get(optimizer_type, Adam(learning_rate=learning_rate))  

                model = self.build_model(input_dim_dict, optimizer=optimizer, **params)  

                model.fit(x_train, 
                          y_train, 
                          validation_data=(x_test, y_test), 
                          epochs=20, 
                          batch_size=batch_size,
                          callbacks=[early_stopping],
                          verbose=0)
                end_time = time.time()
                elapsed_time = end_time - start_time
                loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
                print(f"Accuracy: {accuracy}, Loss: {loss}, Time taken: {elapsed_time:.2f}s")

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                    best_params = params
                    best_optimizer = optimizer

                pbar.update(1)
        
        
        ModelHandler.save_model(best_model, f'{self.model_file_path}/DeepFM_2_16dims.pkl')
        print(f"Best Params: {best_params}")
        print(f"Best Accuracy: {best_accuracy}")
        print(f'Best optimizer: {best_optimizer}"')
        print(f"Model saved to {self.model_file_path}")
        
        final_model = self.retrain_with_best_params(input_dim_dict, best_params, best_optimizer)
        self.plot_metrics(best_model.history, best_loss=loss)

    def retrain_with_best_params(self, input_dim_dict, best_params, best_optimizer):
        print(f"Retraining with best parameters: {best_params}")
        model = self.build_model(input_dim_dict, best_optimizer, **best_params)
        model.compile(
            optimizer=best_optimizer, 
            loss='binary_crossentropy', 
            metrics=['accuracy']
        )

        for i in range(5):  # 5번 반복
            print(f"=== Training Phase {i + 1} ===")
            history = model.fit(
                self.x_train,
                self.y_train,
                validation_data=(self.x_test, self.y_test),
                epochs=20,  # 각 호출에서 20 에포크씩 학습
                batch_size=32,
                verbose=1
            )
            
        # 학습 후 성능 평가
        loss, accuracy = model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"After Phase {i + 1}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")

        self.plot_metrics(history, loss=loss, accuracy=accuracy)

        return model

    def plot_metrics(self, history, best_loss=None, best_accuracy=None):
        # Plot loss
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        if best_loss is not None:
            plt.title(f"Loss (Best: {best_loss:.4f})")
        else:
            plt.title("Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        if best_accuracy is not None:
            plt.title(f"Accuracy (Best: {best_accuracy:.4f})")
        else:
            plt.title("Accuracy")
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()



    def run(self):
            modelhandler = DeepFM_2()
            modelhandler.train_and_save_model()

if __name__ == "__main__":

    deepfm = DeepFM_2()
    deepfm.run()