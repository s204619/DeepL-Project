import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Argument parser for NRMSModel training"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default=str("~/ebnerd_data"),
        help="Path to the data directory",
    )

    # General settings
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument(
        "--datasplit", type=str, default="ebnerd_small", help="Dataset split to use"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    # Batch sizes
    parser.add_argument(
        "--bs_train", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--bs_test", type=int, default=32, help="Batch size for testing"
    )
    parser.add_argument(
        "--batch_size_test_wo_b",
        type=int,
        default=32,
        help="Batch size for testing without balancing",
    )
    parser.add_argument(
        "--batch_size_test_w_b",
        type=int,
        default=4,
        help="Batch size for testing with balancing",
    )

    # History and ratios
    parser.add_argument(
        "--history_size", type=int, default=20, help="History size for the model"
    )
    parser.add_argument(
        "--npratio", type=int, default=4, help="Negative-positive ratio"
    )

    # Training settings
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=1.0,
        help="Fraction of training data to use",
    )
    # parser.add_argument(
    #     "--fraction_test",
    #     type=float,
    #     default=1.0,
    #     help="Fraction of testing data to use",
    # )

    # Model and loader settings
    # parser.add_argument(
    #     "--nrms_loader",
    #     type=str,
    #     default="NRMSDataLoaderPretransform",
    #     choices=["NRMSDataLoaderPretransform", "NRMSDataLoader"],
    #     help="Data loader type (speed or memory efficient)",
    # )

    # # Chunk processing
    # parser.add_argument(
    #     "--n_chunks_test", type=int, default=10, help="Number of test chunks to process"
    # )
    # parser.add_argument(
    #     "--chunks_done", type=int, default=0, help="Number of chunks already processed"
    # )

    # =====================================================================================
    #  ############################# UNIQUE FOR NRMSDocVec ###############################
    # =====================================================================================
    # Transformer settings
    parser.add_argument(
        "--transformer_model_name",
        type=str,
        default="FacebookAI/xlm-roberta-large",
        help="Transformer model name",
    )
    parser.add_argument(
        "--max_title_length",
        type=int,
        default=30,
        help="Maximum length of title encoding",
    )

    # Hyperparameters
    parser.add_argument(
        "--head_num", type=int, default=20, help="Number of attention heads"
    )
    parser.add_argument(
        "--head_dim", type=int, default=20, help="Dimension of each attention head"
    )
    parser.add_argument(
        "--attention_hidden_dim",
        type=int,
        default=200,
        help="Dimension of attention hidden layers",
    )

    # Optimizer settings
    parser.add_argument(
        "--optimizer", type=str, default="adam", help="Optimizer to use"
    )
    parser.add_argument(
        "--loss", type=str, default="cross_entropy_loss", help="Loss function"
    )
    parser.add_argument("--dropout", type=float, default=0.20, help="Dropout rate")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )

    return parser.parse_args()
