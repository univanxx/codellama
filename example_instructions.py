# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

import fire

from llama import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_seq_len: int = 4096,
    max_batch_size: int = 1,
    max_gen_len: Optional[int] = None,
    local_rank: int=0,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    instructions = [
        [
            {"role": "user", "content": """Tell me with steps and code how to create Telegram bot which takes text input, goes to SSH server, where
             odels runs on gpu, uses it as input and then takes model output and return as bot answer."""
             },
        ]

#         [
#             {"role": "user", "content": "How to make a xgboost model whick takes output from mediapipe library from one photo and predict scalar?"},
#             {
#                 "role": "assistant",
#                 "content": """\
# To make a XGBoost model that takes output from the MediaPipe library from one photo and predicts a scalar, you can follow these steps:

# 1. Install the necessary libraries:
# ```
# pip install mediapipe
# pip install xgboost
# ```
# 2. Load the MediaPipe library and the XGBoost library:
# ```
# import mediapipe as mp
# import xgboost as xgb
# ```
# 3. Load the image and extract the facial landmarks using the MediaPipe library:
# ```
# image = mp.read_image("image.jpg")
# landmarks = mp.get_facial_landmarks(image)
# ```""",
#             },
#             {"role": "user", "content": "Step 3 outputs: AttributeError: module 'mediapipe' has no attribute 'read_image' and 'get_facial_landmarks'. Fix this please and make sure you are not hallucinating."},
#         ]
    ]
        # [
        #     {
        #         "role": "system",
        #         "content": "Provide text message with explanations and then detailed code in Python",
        #     },
        #     {
        #         "role": "user",
        #         "content": "How to write telegram bot which can accept a text request and pass it to a shell script from the server on gpu A100? Script returns text",
        #     }
        # ],
        # [
            # {
            #     "role": "system",
            #     "content": "Provide text message with explanations and then detailed code in Python",
            # },
        #     {
        #         "role": "user",
        #         "content": "How to make codellama remember previous dialogue? Write code example for the Instruct llama if possible",
        #     }
        # ],
        # [
        #     {
        #         "role": "user",
        #         "content": "What is the difference between inorder and preorder traversal? Give an example in Python.",
        #     }
        # ],
        # [
        #     {
        #         "role": "system",
        #         "content": "Provide answers in JavaScript",
        #     },
        #     {
        #         "role": "user",
        #         "content": "Write a function that computes the set of sums of all contiguous sublists of a given list.",
        #     }
        # ],
    # ]
    results = generator.chat_completion(
        instructions,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for instruction, result in zip(instructions, results):
        for msg in instruction:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
