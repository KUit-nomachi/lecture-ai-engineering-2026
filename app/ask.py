"""第1回: 素のLLMに会話履歴を保ちながら質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールは使わず、同じチャットで会話履歴を保ちながら質問する。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

#変更点_課題_基礎
from google.genai import types

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(chat, question: str) -> str:
    response = chat.send_message(question)

    usage = response.usage_metadata
    if usage is not None:
        print(
            f"[usage] input_tokens={usage.prompt_token_count} "
            f"output_tokens={usage.candidates_token_count}"
        )

    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="生成時のランダム性を調整するパラメータ",
    )
    args = parser.parse_args()

    client = genai.Client(api_key=get_api_key())
    config = types.GenerateContentConfig(temperature=args.temperature)
    chat = client.chats.create(model=MODEL, config=config)

    print(f"[model] {MODEL}")
    print(f"[temperature] {args.temperature}\n")
    try:
        question = args.question
        while True:
            print(f"[質問] {question}\n")
            print(ask(chat, question))
            try:
                question = input("\n続けて質問してください（終了: 空行または q）: ").strip()
            except EOFError:
                print()
                break
            if not question or question.lower() in {"q", "quit", "exit"}:
                break
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
