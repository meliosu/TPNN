import numpy as np
import pandas as pd


def preprocess(students: pd.DataFrame) -> pd.DataFrame:
    variables = {
        'binary': [
            'school',
            'sex',
            'address',
            'famsize',
            'Pstatus',
            'schoolsup',
            'famsup',
            'paid',
            'activities',
            'nursery',
            'higher',
            'internet',
            'romantic',
        ],

        'categorical': [
            'Mjob',
            'Fjob',
            'reason',
            'guardian',
        ],

        'numeric': {
            'age': (15, 22),
            'Medu': (0, 4),
            'Fedu': (0, 4),
            'traveltime': (0, 4),
            'studytime': (0, 4),
            'failures': (1, 4),
            'famrel': (1, 5),
            'freetime': (1, 5),
            'goout': (1, 5),
            'Dalc': (1, 5),
            'Walc': (1, 5),
            'health': (1, 5),
            'absences': (0, 93),
            'G1': (0, 20),
            'G2': (0, 20),
            'G3': (0, 20),
        },
    }

    preprocessed = pd.DataFrame()

    for variable in variables['binary']:
        unique = students[variable].unique()
        mapping = {unique[0]: 0.0, unique[1]: 1.0}
        preprocessed[variable] = students[variable].map(mapping)

    for variable in variables['categorical']:
        onehot = pd.get_dummies(students[variable], dtype=float)

        for col in onehot:
            preprocessed[col] = onehot[col]

    for variable, bounds in variables['numeric'].items():
        lower = bounds[0]
        higher = bounds[1]
        scaled = students[variable].map(lambda x: (x - lower) / (higher - lower)).astype(float)
        preprocessed[variable] = scaled

    return preprocessed


def data() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    test_ratio = 0.2

    students = pd.read_csv('../dataset/students.csv', sep=';')
    preprocessed = preprocess(students)

    array = preprocessed.to_numpy()

    np.random.seed(42)
    np.random.shuffle(array)

    x, y = array[:, :-1], array[:, -1]

    samples = array.shape[0]
    test_size = int(samples * test_ratio)

    x_train, x_test = x[test_size:], x[:test_size]
    y_train, y_test = y[test_size:], y[:test_size]

    return (x_train, y_train), (x_test, y_test)
