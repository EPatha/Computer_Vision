import os
import chess.pgn
from cairosvg import svg2png
from PIL import Image
import chess.svg

def generate_fen_diagrams(pgn_file_path, output_directory, image_format="PNG"):
    valid_formats = {"PNG", "JPEG"}
    image_format = image_format.upper()
    if image_format not in valid_formats:
        raise ValueError(f"Format gambar '{image_format}' tidak valid. Gunakan salah satu dari: {valid_formats}")

    with open(pgn_file_path) as pgn_file:
        game = chess.pgn.read_game(pgn_file)

    if not game:
        raise ValueError(f"Tidak ada game dalam file {pgn_file_path}")

    board = game.board()
    fen_list = []

    for i, move in enumerate(game.mainline_moves(), start=1):
        board.push(move)
        fen_list.append(board.fen())
        print(f"Step {i}: {board.fen()}")

    os.makedirs(output_directory, exist_ok=True)

    for i, fen in enumerate(fen_list, start=1):
        board_svg = chess.svg.board(chess.Board(fen))
        svg_path = os.path.join(output_directory, f"step_{i}.svg")
        png_path = os.path.join(output_directory, f"step_{i}.{image_format.lower()}")

        with open(svg_path, "w") as svg_file:
            svg_file.write(board_svg)

        svg2png(bytestring=board_svg, write_to=png_path)

        if image_format == "JPEG":
            img = Image.open(png_path)
            jpeg_path = os.path.join(output_directory, f"step_{i}.jpeg")
            img.convert("RGB").save(jpeg_path, format="JPEG")
            os.remove(png_path)
            print(f"Step {i}: {fen} -> Saved as {jpeg_path}")
        else:
            print(f"Step {i}: {fen} -> Saved as {png_path}")


if __name__ == "__main__":
    pgn_file_path = "/home/ep/Documents/Github/Computer_Vision/Klasifikasi_Chess/data/train.pgn"  # Path file PGN
    output_directory = "/home/ep/Documents/Github/Computer_Vision/Klasifikasi_Chess/data/FEN"     # Path folder output
    image_format = "PNG"  # Pilih PNG atau JPEG

    generate_fen_diagrams(pgn_file_path, output_directory, image_format)
