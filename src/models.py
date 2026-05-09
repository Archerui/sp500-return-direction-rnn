import torch
import torch.nn as nn


class RecurrentClassifier(nn.Module):
    """
    A shared classifier for RNN, LSTM, and GRU.

    Input shape:
        x: [batch_size, sequence_length, input_dim]

    Output shape:
        logits: [batch_size, num_classes]
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        num_layers=1,
        num_classes=3,
        rnn_type="LSTM",
        dropout=0.2,
        bidirectional=False,
    ):
        super().__init__()

        rnn_type = rnn_type.upper()

        if rnn_type == "RNN":
            rnn_cls = nn.RNN
        elif rnn_type == "LSTM":
            rnn_cls = nn.LSTM
        elif rnn_type == "GRU":
            rnn_cls = nn.GRU
        else:
            raise ValueError("rnn_type must be one of: RNN, LSTM, GRU")

        self.rnn_type = rnn_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        direction_multiplier = 2 if bidirectional else 1

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * direction_multiplier, num_classes),
        )

    def forward(self, x):
        """
        x shape:
            [batch_size, sequence_length, input_dim]
        """
        output, hidden = self.rnn(x)

        # output shape:
        # [batch_size, sequence_length, hidden_dim * num_directions]
        last_output = output[:, -1, :]

        logits = self.classifier(last_output)

        return logits


def count_parameters(model):
    """
    Count trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(
    model_type,
    input_dim,
    hidden_dim=64,
    num_layers=1,
    num_classes=3,
    dropout=0.2,
    bidirectional=False,
):
    """
    Factory function for building RNN/LSTM/GRU models.
    """
    return RecurrentClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        rnn_type=model_type,
        dropout=dropout,
        bidirectional=bidirectional,
    )


if __name__ == "__main__":
    batch_size = 16
    sequence_length = 30
    input_dim = 13
    num_classes = 3

    x = torch.randn(batch_size, sequence_length, input_dim)

    for model_type in ["RNN", "LSTM", "GRU"]:
        model = build_model(
            model_type=model_type,
            input_dim=input_dim,
            hidden_dim=64,
            num_layers=1,
            num_classes=num_classes,
            dropout=0.2,
        )

        logits = model(x)

        print("=" * 80)
        print(model_type)
        print("=" * 80)
        print(model)
        print("Input shape:", x.shape)
        print("Output shape:", logits.shape)
        print("Trainable parameters:", count_parameters(model))