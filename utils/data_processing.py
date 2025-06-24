import pandas as pd
import numpy as np

def read_and_split_csv(file_path):
    data = pd.read_csv(file_path)
    print(f"读取的文件 {file_path} 的形状: {data.shape}")
    train_data = data[data['group'] == 'train']
    test_data = data[data['group'] == 'test']
    print(f"训练集的形状: {train_data.shape}, 测试集的形状: {test_data.shape}")
    return train_data, test_data

def read_all_features(file_paths):
    all_train_x = []
    all_test_x = []
    used_features = {}
    for file_path in file_paths:
        train_data, test_data = read_and_split_csv(file_path)
        train_x = train_data.drop(['group', 'label', 'smiles'], axis=1) if 'smiles' in train_data.columns else train_data.drop(['group', 'label'], axis=1)
        test_x = test_data.drop(['group', 'label', 'smiles'], axis=1) if 'smiles' in test_data.columns else test_data.drop(['group', 'label'], axis=1)
        used_features[file_path] = train_x.columns.tolist()
        all_train_x.append(train_x)
        all_test_x.append(test_x)
        print(f"从文件 {file_path} 提取的训练特征形状: {train_x.shape}, 测试特征形状: {test_x.shape}")
    all_train_x = pd.concat(all_train_x, axis=1)
    all_test_x = pd.concat(all_test_x, axis=1)
    print(f"拼接后的训练特征形状: {all_train_x.shape}, 测试特征形状: {all_test_x.shape}")
    return all_train_x, all_test_x, used_features

def read_supplementary_features(supplementary_file_paths):
    all_train_supplementary_x = []
    all_test_supplementary_x = []
    used_features = {}
    for file_path in supplementary_file_paths:
        train_data, test_data = read_and_split_csv(file_path)
        train_supplementary_x = train_data.drop(['group', 'label', 'smiles'], axis=1) if 'smiles' in train_data.columns else train_data.drop(['group', 'label'], axis=1)
        test_supplementary_x = test_data.drop(['group', 'label', 'smiles'], axis=1) if 'smiles' in test_data.columns else test_data.drop(['group', 'label'], axis=1)
        used_features[file_path] = train_supplementary_x.columns.tolist()
        all_train_supplementary_x.append(train_supplementary_x)
        all_test_supplementary_x.append(test_supplementary_x)
        print(f"从补充文件 {file_path} 提取的训练特征形状: {train_supplementary_x.shape}, 测试特征形状: {test_supplementary_x.shape}")
    all_train_supplementary_x = pd.concat(all_train_supplementary_x, axis=1)
    all_test_supplementary_x = pd.concat(all_test_supplementary_x, axis=1)
    print(f"拼接后的补充训练特征形状: {all_train_supplementary_x.shape}, 测试特征形状: {all_test_supplementary_x.shape}")
    return all_train_supplementary_x, all_test_supplementary_x, used_features

def feature_concat(train_x, train_supplementary_x, test_x, test_supplementary_x):
    train_all_features = pd.concat([train_x.reset_index(drop=True), train_supplementary_x.reset_index(drop=True)], axis=1)
    test_all_features = pd.concat([test_x.reset_index(drop=True), test_supplementary_x.reset_index(drop=True)], axis=1)
    print(f"特征拼接后训练集特征形状: {train_all_features.shape}, 测试集特征形状: {test_all_features.shape}")
    return train_all_features, test_all_features

def data_augmentation(X, y, num_augmented_samples=10):
    augmented_X = []
    augmented_y = []
    for _ in range(num_augmented_samples):
        noise = np.random.normal(0, 0.01, X.shape)
        augmented_X.append(X + noise)
        augmented_y.extend(y)
    augmented_X = np.concatenate(augmented_X, axis=0)
    augmented_y = np.array(augmented_y)
    return augmented_X, augmented_y

def contrastive_learning(X, y, num_negative_samples=5):
    positive_X = []
    positive_y = []
    negative_X = []
    negative_y = []
    for i in range(len(X)):
        positive_X.append(X[i])
        positive_y.append(y[i])
        for _ in range(num_negative_samples):
            random_index = np.random.choice(len(X))
            while random_index == i:
                random_index = np.random.choice(len(X))
            negative_X.append(X[random_index])
            negative_y.append(y[random_index])
    new_X = np.concatenate([np.array(positive_X), np.array(negative_X)], axis=0)
    new_y = np.concatenate([np.array(positive_y), np.array(negative_y)], axis=0)
    return new_X, new_y