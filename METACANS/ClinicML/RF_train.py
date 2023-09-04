import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from utils_RF import make_df, test_and_print
import argparse

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_data', type=str,
                        default='/media/data1/doohyun/wsi/code/RF/excel')
    parser.add_argument('--path_id', type=str,
                        default='/media/data1/doohyun/wsi/code/MIL_base/code_and_model/results')
    return parser


def main_worker(args):
    X_train, y_train, feature_names = make_df(args.path_data, args.path_id, 'ss_tr')
    
    RFC = RandomForestClassifier(n_estimators=2000,
                                 max_depth=3,
                                 min_samples_split=50,
                                 class_weight='balanced',
                                 random_state=42)
    RFC.fit(X_train, y_train)
    importances = RFC.feature_importances_
    sorted_indices = np.argsort(importances)
    sorted_feature_names = [feature_names[i] for i in sorted_indices]
    sorted_importances = importances[sorted_indices]
    print("Sorted feature importances:")
    for feature, importance in zip(sorted_feature_names, sorted_importances):
        print(f"{feature}: {importance}")


    ''''''
    test_and_print(X_train, y_train, 'ss_tr', RFC)
    
    X_valid, y_valid, _ = make_df(args.path_data, args.path_id, 'ss_val')
    test_and_print(X_valid, y_valid, 'ss_val', RFC)
    X_valid, y_valid, _ = make_df(args.path_data, args.path_id, 'ss_te')
    test_and_print(X_valid, y_valid, 'ss_te', RFC)
    X_valid, y_valid, _ = make_df(args.path_data, args.path_id, 'km')
    test_and_print(X_valid, y_valid, 'km', RFC)
    X_valid, y_valid, _ = make_df(args.path_data, args.path_id, 'gs')
    test_and_print(X_valid, y_valid, 'gs', RFC)
    X_valid, y_valid, _ = make_df(args.path_data, args.path_id, 'ewha')
    test_and_print(X_valid, y_valid, 'ewha', RFC)
    X_valid, y_valid, _ = make_df(args.path_data, args.path_id, 'gc')
    test_and_print(X_valid, y_valid, 'gc', RFC)
    X_valid, y_valid, _ = make_df(args.path_data, args.path_id, 'dk')
    test_and_print(X_valid, y_valid, 'dk', RFC)

    plt.figure()
    plt.title("Feature importances")
    plt.barh(range(X_train.shape[1]), sorted_importances, align='center')
    plt.yticks(range(X_train.shape[1]), sorted_feature_names)
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.show()
    plt.savefig('feature_importances.png', bbox_inches='tight', dpi=600)
    

if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    main_worker(args)