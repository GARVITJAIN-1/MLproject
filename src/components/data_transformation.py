"""
src/components/data_transformation.py
-------------------------------------
Creates and applies a preprocessing pipeline:

• Numerical columns  → median impute → standard-scale  
• Categorical columns→ most-freq impute → one-hot encode (dense)  
  (scaling one-hot vectors is not necessary, so we skip it)

The fitted ColumnTransformer is saved to artifacts/preprocessor.pkl
and the function returns the transformed train/test arrays plus the
path to the saved object.
"""

import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# --------------------------------------------------------------------------- #
# Configuration dataclass
# --------------------------------------------------------------------------- #

@dataclass
class DataTransformationConfig:
    """Stores configuration values for this component."""
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


# --------------------------------------------------------------------------- #
# Main class
# --------------------------------------------------------------------------- #

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    # --------------------------------------------------------------------- #
    # Build the ColumnTransformer
    # --------------------------------------------------------------------- #
    @staticmethod
    def get_data_transformer_object() -> ColumnTransformer:
        """
        Constructs and returns a preprocessing pipeline.

        Returns
        -------
        ColumnTransformer
            Fitted later to transform both training and test sets.
        """
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Numerical pipeline: median impute → standard scale
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            # Categorical pipeline: most-freq impute → one-hot (dense)
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "one_hot_encoder",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            )

            logging.info(f"Numerical columns   : {numerical_columns}")
            logging.info(f"Categorical columns : {categorical_columns}")

            # Combine both pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys) from e

    # --------------------------------------------------------------------- #
    # Apply the transformer and save it
    # --------------------------------------------------------------------- #
    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Reads CSVs, fits the preprocessor, transforms data,
        saves the preprocessor, and returns transformed arrays.

        Parameters
        ----------
        train_path : str
            Path to training CSV.
        test_path : str
            Path to test CSV.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, str]
            Transformed train array, test array, and path to saved preprocessor.
        """
        try:
            # Load data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Loaded training and test data.")

            # Build preprocessor
            preprocessor = self.get_data_transformer_object()

            target_column_name = "math_score"

            # Split features / target
            X_train = train_df.drop(columns=[target_column_name])
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=[target_column_name])
            y_test = test_df[target_column_name]

            # Fit / transform
            logging.info("Fitting preprocessing pipeline on training data.")
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            # Concatenate target column back for downstream components
            train_arr = np.c_[X_train_processed, y_train.values]
            test_arr = np.c_[X_test_processed, y_test.values]

            # Persist the fitted preprocessor
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor,
            )
            logging.info(
                f"Preprocessing object saved to "
                f"{self.data_transformation_config.preprocessor_obj_file_path}"
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys) from e
