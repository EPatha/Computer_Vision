import os
import chess.pgn
from cairosvg import svg2png
from PIL import Image
import chess.svg

def generate_fen_diagrams_for_all_games(pgn_file_path, output_directory, image_format="PNG"):
    valid_formats = {"PNG", "JPEG"}
    image_format = image_format.upper()
    if image_format not in valid_formats:
        raise ValueError(f"Format gambar '{image_format}' tidak valid. Gunakan salah satu dari: {valid_formats}")

    # Pastikan direktori output ada
    os.makedirs(output_directory, exist_ok=True)

    with open(pgn_file_path) as pgn_file:
        game_index = 1
        while True:
            # Membaca game berikutnya dalam file PGN
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break  # Tidak ada game lagi

            board = game.board()
            fen_list = []

            # Proses langkah-langkah dalam game
            for move_index, move in enumerate(game.mainline_moves(), start=1):
                board.push(move)
                fen_list.append(board.fen())

            # Buat folder untuk game ini
            game_output_dir = os.path.join(output_directory, f"game_{game_index}")
            os.makedirs(game_output_dir, exist_ok=True)

            # Simpan diagram langkah-langkah game ini
            for step_index, fen in enumerate(fen_list, start=1):
                board_svg = chess.svg.board(chess.Board(fen))
                svg_path = os.path.join(game_output_dir, f"step_{step_index}.svg")
                image_path = os.path.join(game_output_dir, f"step_{step_index}.{image_format.lower()}")

                # Simpan file SVG
                with open(svg_path, "w") as svg_file:
                    svg_file.write(board_svg)

                # Konversi ke format gambar (PNG atau JPEG)
                svg2png(bytestring=board_svg, write_to=image_path)

                if image_format == "JPEG":
                    # Jika format JPEG, konversi dan hapus PNG
                    img = Image.open(image_path)
                    jpeg_path = os.path.join(game_output_dir, f"step_{step_index}.jpeg")
                    img.convert("RGB").save(jpeg_path, format="JPEG")
                    os.remove(image_path)  # Hapus PNG
                    print(f"Game {game_index}, Step {step_index}: {fen} -> Saved as {jpeg_path}")
                else:
                    print(f"Game {game_index}, Step {step_index}: {fen} -> Saved as {image_path}")

            game_index += 1  # Pindah ke game berikutnya


if __name__ == "__main__":
    pgn_file_path = "/home/ep/Documents/Github/Computer_Vision/Klasifikasi_Chess/data/train.pgn"  # Path file PGN
    output_directory = "/home/ep/Documents/Github/Computer_Vision/Klasifikasi_Chess/data/image"     # Path folder output
    image_format = "PNG"  # Pilih PNG atau JPEG

    generate_fen_diagrams_for_all_games(pgn_file_path, output_directory, image_format)
