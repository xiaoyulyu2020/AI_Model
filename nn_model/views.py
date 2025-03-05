from typing import Any

import torch
from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework.decorators import action
from .nn_model_app import LinearModel
from .nn_train import MODEL_SAVE_PATH, ai_model_train


class PredictionViewSet(viewsets.ViewSet):
    def __init__(self, **kwargs: Any):
        """Load the trained model during initialization."""
        super().__init__()
        self.model_nn = LinearModel()
        self.model_nn.load_state_dict(torch.load(MODEL_SAVE_PATH))
        self.model_nn.eval()
    @action(detail=False, methods=['post'])
    def predict(self, request):
        """
        A custom action to handle prediction.
        This expects `user_input` in the request body.
        """
        try:
            # Extract user input from request data
            user_input = request.data.get('user_input', None)

            if user_input is None:
                return Response({'error': 'User input is required'}, status=400)

            # Convert input to tensor (ensure it's a float)
            user_input_tensor = torch.tensor([[float(user_input)]], dtype=torch.float32)  # Reshape to (1,1)

            # Get prediction
            with torch.no_grad():
                prediction = self.model_nn(user_input_tensor).item()  # Convert tensor output to float

            return Response({'prediction': prediction}, status=200)

        except Exception as e:
            return Response({'error': str(e)}, status=500)