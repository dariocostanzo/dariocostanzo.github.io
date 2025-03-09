
let board = null;
let game = new Chess();
let moveHistory = [];

function onDragStart(source, piece, position, orientation) {
    // Don't allow moving pieces if the game is over
    if (game.game_over()) return false;
    
    // Only allow white pieces to be moved
    if (piece.search(/^b/) !== -1) return false;
}

function makeRandomMove() {
    const possibleMoves = game.moves();
    
    // Game over
    if (possibleMoves.length === 0) return;
    
    const randomIdx = Math.floor(Math.random() * possibleMoves.length);
    game.move(possibleMoves[randomIdx]);
    board.position(game.fen());
    updateStatus();
}

function makeAIMove() {
    const difficulty = document.getElementById('difficulty').value;
    document.getElementById('status').textContent = 'AI is thinking...';
    
    // Send the current board position to the server
    fetch('/api/move', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            fen: game.fen(),
            difficulty: difficulty
        }),
    })
    .then(response => response.json())
    .then(data => {
        // Make the move returned by the AI
        game.move(data.move);
        board.position(game.fen());
        
        // Record the move
        moveHistory.push(data.move);
        updateHistory();
        
        // Update the game status
        updateStatus();
    })
    .catch(error => {
        console.error('Error:', error);
        // Fallback to random move if the server fails
        makeRandomMove();
    });
}

function onDrop(source, target) {
    // See if the move is legal
    const move = game.move({
        from: source,
        to: target,
        promotion: 'q' // Always promote to a queen for simplicity
    });
    
    // Illegal move
    if (move === null) return 'snapback';
    
    // Record the move
    moveHistory.push(move.san);
    updateHistory();
    
    // Update the board position
    updateStatus();
    
    // Make the AI move after a short delay
    setTimeout(makeAIMove, 250);
}

function updateHistory() {
    const historyElement = document.getElementById('history');
    let historyHTML = '';
    
    for (let i = 0; i < moveHistory.length; i += 2) {
        const moveNumber = Math.floor(i / 2) + 1;
        const whiteMove = moveHistory[i];
        const blackMove = moveHistory[i + 1] ? moveHistory[i + 1] : '';
        
        historyHTML += `${moveNumber}. ${whiteMove} ${blackMove} `;
    }
    
    historyElement.textContent = historyHTML;
}

function updateStatus() {
    let status = '';
    
    if (game.in_checkmate()) {
        status = game.turn() === 'w' ? 'Game over, black wins by checkmate!' : 'Game over, white wins by checkmate!';
    } else if (game.in_draw()) {
        status = 'Game over, drawn position';
    } else {
        status = game.turn() === 'w' ? 'Your turn (White)' : 'AI is thinking...';
        
        if (game.in_check()) {
            status += ', ' + (game.turn() === 'w' ? 'White' : 'Black') + ' is in check';
        }
    }
    
    document.getElementById('status').textContent = status;
}

function newGame() {
    game = new Chess();
    moveHistory = [];
    updateHistory();
    board.position('start');
    updateStatus();
}

document.addEventListener('DOMContentLoaded', function() {
    const config = {
        draggable: true,
        position: 'start',
        onDragStart: onDragStart,
        onDrop: onDrop
    };
    
    board = Chessboard('board', config);
    updateStatus();
    
    // Event listeners for buttons
    document.getElementById('newGame').addEventListener('click', newGame);
    
    document.getElementById('flipBoard').addEventListener('click', function() {
        board.flip();
    });
});
