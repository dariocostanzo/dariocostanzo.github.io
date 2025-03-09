# File: backend/app.py
from flask import Flask, request, jsonify, render_template
from chess_engine import ChessAI, ChessGame
import os

app = Flask(__name__, static_folder='../frontend', template_folder='../frontend')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/move', methods=['POST'])
def make_move():
    data = request.json
    fen = data.get('fen')
    difficulty = data.get('difficulty', 'medium')
    
    game = ChessGame(fen)
    ai = ChessAI(difficulty)
    
    ai_move = ai.get_best_move(game)
    game.make_move(ai_move)
    
    return jsonify({
        'move': ai_move,
        'fen': game.get_fen(),
        'game_over': game.is_game_over(),
        'result': game.get_result() if game.is_game_over() else None
    })

if __name__ == '__main__':
    app.run(debug=True)


# File: backend/chess_engine.py
import chess
import random
import numpy as np
from model import ChessEvaluationModel

class ChessGame:
    def __init__(self, fen=None):
        if fen:
            self.board = chess.Board(fen)
        else:
            self.board = chess.Board()
    
    def get_legal_moves(self):
        return [move.uci() for move in self.board.legal_moves]
    
    def make_move(self, move_uci):
        move = chess.Move.from_uci(move_uci)
        if move in self.board.legal_moves:
            self.board.push(move)
            return True
        return False
    
    def get_fen(self):
        return self.board.fen()
    
    def is_game_over(self):
        return self.board.is_game_over()
    
    def get_result(self):
        if self.board.is_checkmate():
            return "checkmate"
        elif self.board.is_stalemate():
            return "stalemate"
        elif self.board.is_insufficient_material():
            return "insufficient_material"
        elif self.board.is_seventyfive_moves():
            return "seventyfive_moves"
        elif self.board.is_fivefold_repetition():
            return "fivefold_repetition"
        return "draw"

class ChessAI:
    def __init__(self, difficulty='medium'):
        self.difficulty = difficulty
        self.model = ChessEvaluationModel()
        self.max_depth = {
            'easy': 2,
            'medium': 3,
            'hard': 4
        }.get(difficulty, 3)
    
    def get_best_move(self, game):
        if self.difficulty == 'easy' and random.random() < 0.3:
            # 30% chance to make a random move in easy mode
            return random.choice(game.get_legal_moves())
        
        best_score = float('-inf')
        best_move = None
        
        for move in game.get_legal_moves():
            game_copy = ChessGame(game.get_fen())
            game_copy.make_move(move)
            
            score = -self._minimax(game_copy, self.max_depth - 1, float('-inf'), float('inf'), False)
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def _minimax(self, game, depth, alpha, beta, maximizing_player):
        if depth == 0 or game.is_game_over():
            return self._evaluate_position(game)
        
        if maximizing_player:
            max_eval = float('-inf')
            for move in game.get_legal_moves():
                game_copy = ChessGame(game.get_fen())
                game_copy.make_move(move)
                eval = self._minimax(game_copy, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in game.get_legal_moves():
                game_copy = ChessGame(game.get_fen())
                game_copy.make_move(move)
                eval = self._minimax(game_copy, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
    
    def _evaluate_position(self, game):
        # Basic material counting for fallback
        board = game.board
        
        # First try to use the neural network model
        try:
            board_tensor = self._board_to_tensor(board)
            return self.model.predict(board_tensor)[0]
        except:
            # Fallback to basic material counting
            if board.is_checkmate():
                return -10000 if board.turn else 10000
            
            material_value = 0
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece:
                    value = {
                        chess.PAWN: 1,
                        chess.KNIGHT: 3,
                        chess.BISHOP: 3,
                        chess.ROOK: 5,
                        chess.QUEEN: 9,
                        chess.KING: 0  # King has no material value in this simple eval
                    }[piece.piece_type]
                    
                    if piece.color == chess.WHITE:
                        material_value += value
                    else:
                        material_value -= value
            
            return material_value
    
    def _board_to_tensor(self, board):
        # Convert chess board to a format suitable for the neural network
        tensor = np.zeros((8, 8, 12), dtype=np.float32)
        
        piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank, file = chess.square_rank(square), chess.square_file(square)
                piece_idx = piece_types.index(piece.piece_type)
                if piece.color == chess.BLACK:
                    piece_idx += 6
                tensor[rank, file, piece_idx] = 1
        
        return tensor.reshape(1, -1)


# File: model.py
import numpy as np
import os
import pickle

class ChessEvaluationModel:
    def __init__(self, model_path=None):
        self.model_path = model_path or os.path.join(os.path.dirname(__file__), '../models/chess_model.pkl')
        self.model = self._load_model() if os.path.exists(self.model_path) else self._create_simple_model()
    
    def _load_model(self):
        try:
            with open(self.model_path, 'rb') as f:
                return pickle.load(f)
        except:
            return self._create_simple_model()
    
    def _create_simple_model(self):
        # A very basic model for demonstration - in a real project you'd use PyTorch/TensorFlow here
        class SimpleModel:
            def __init__(self):
                # Initialize with basic piece weights (pawns, knights, bishops, rooks, queens, kings) 
                # for white and black (12 values) + positional bonuses for center control
                self.piece_weights = np.array([1, 3, 3, 5, 9, 0, -1, -3, -3, -5, -9, 0])
                
                # Center squares are more valuable
                self.position_bonus = np.zeros((8, 8))
                self.position_bonus[3:5, 3:5] = 0.3  # Center
                self.position_bonus[2:6, 2:6] = 0.1  # Extended center
            
            def predict(self, board_tensor):
                # Reshape if needed
                if board_tensor.ndim == 1:
                    board_tensor = board_tensor.reshape(8, 8, 12)
                
                # Material evaluation
                material_sum = np.sum(board_tensor.reshape(-1, 12) * self.piece_weights, axis=1)
                
                # Position evaluation (simplified)
                position_value = 0
                for piece_idx in range(12):
                    piece_positions = board_tensor[:, :, piece_idx]
                    # Positive for white pieces, negative for black pieces
                    if piece_idx < 6:
                        position_value += np.sum(piece_positions * self.position_bonus)
                    else:
                        position_value -= np.sum(piece_positions * self.position_bonus)
                
                return np.array([material_sum + position_value])
        
        return SimpleModel()
    
    def predict(self, board_tensor):
        return self.model.predict(board_tensor)

# We'd also want to train this model, but for this example we're just using a simple heuristic model


# Frontend code: frontend/index.html
'''

'''