import os
import chess.pgn
from pathlib import Path

def classify_games_by_opening(pgn_file_path, output_directory):
    # Buka file PGN
    with open(pgn_file_path, "r") as pgn_file:
        game_number = 0

        # Iterasi melalui setiap game dalam file PGN
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break  # Jika tidak ada game lagi, keluar dari loop

            game_number += 1

            # Ambil nama pembukaan dari metadata
            opening_name = game.headers.get("ECOUrl", "Unknown").split("/")[-1]
            if opening_name == "Unknown":
                print(f"Game {game_number}: Pembukaan tidak ditemukan, melewati.")
                continue

            # Bersihkan nama direktori (ganti karakter tidak valid dengan "_")
            safe_opening_name = "".join(
                [c if c.isalnum() or c in " -_" else "_" for c in opening_name]
            )

            # Tentukan direktori untuk pembukaan ini
            opening_dir = os.path.join(output_directory, safe_opening_name)
            os.makedirs(opening_dir, exist_ok=True)

            # Buat subdirektori untuk game ini
            game_dir = os.path.join(opening_dir, f"game_{game_number}")
            os.makedirs(game_dir, exist_ok=True)

            # Iterasi langkah-langkah dan simpan FEN untuk setiap langkah
            board = game.board()
            for step, move in enumerate(game.mainline_moves(), start=1):
                board.push(move)
                fen = board.fen()
                fen_file = os.path.join(game_dir, f"step_{step}.fen")
                with open(fen_file, "w") as fen_output:
                    fen_output.write(fen)

            print(f"Game {game_number} disimpan di {game_dir}")

# Contoh penggunaan
pgn_file_path = "/home/ep/Documents/Github/Computer_Vision/Klasifikasi_Chess/data/train.pgn"  # Ganti dengan path ke file PGN Anda
output_directory = "/home/ep/Documents/Github/Computer_Vision/Klasifikasi_Chess/data/FEN/"
classify_games_by_opening(pgn_file_path, output_directory)
