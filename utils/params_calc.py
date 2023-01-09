def calculate_num_parameters(num_heads, head_dim, num_layers, vocab_size, context_length,
                             multi_group=0, with_embeddings=True):
    # accurate for gpt3 family
    # not accurate for palm -- for palm, we might need to take into account SwiGLU
    num_parameters = 0
    hidden_dim = num_heads * head_dim
    # projection Q,K,V,O is dxd each for multi head
    # for MG, K and V are d x g*k (h*k=d, g<h)
    if multi_group > 0:
        num_parameters += ((hidden_dim + 1) * (2*hidden_dim + 2*multi_group*head_dim)) * num_layers
    else:
        num_parameters += (hidden_dim + 1) * hidden_dim * 4 * num_layers
    # linear layer is dx4d and 4dxd
    num_parameters += ((hidden_dim + 1) * hidden_dim * 4 + 4 * hidden_dim * hidden_dim + hidden_dim) * num_layers
    # layer norm params for weight and bias and 2 layers of LN
    num_parameters += hidden_dim * 2 * 2 * num_layers
    # weight and bias of layer norm final
    num_parameters += hidden_dim * 2

    if with_embeddings:
        # final embedding layer
        num_parameters += (vocab_size + 1) * hidden_dim
        # position embeddings
        num_parameters += context_length * hidden_dim
    return num_parameters / 1e9


def estimate_gpt3():
    print("GPT3 2.7B",
          calculate_num_parameters(num_heads=32, head_dim=2560//32, num_layers=32, vocab_size=50257, context_length=2048,
                                   multi_group=0,
                                   )
          )
    print("GPT3 6.7B",
          calculate_num_parameters(num_heads=32, head_dim=4096 // 32, num_layers=32, vocab_size=50257,
                                   context_length=2048,
                                   multi_group=0,
                                   )
          )
    print("GPT3 13B",
          calculate_num_parameters(num_heads=40, head_dim=5140 // 40, num_layers=40, vocab_size=50257,
                                   context_length=2048,
                                   multi_group=0,
                                   )
          )
    print("GPT3 175B",
          calculate_num_parameters(num_heads=96, head_dim=12288 // 96, num_layers=96, vocab_size=50257,
                                   context_length=2048,
                                   multi_group=0,
                                   )
          )

def estimate_bloom():
    print("Bloom 175B",
          calculate_num_parameters(num_heads=112, head_dim=128, num_layers=70, vocab_size=50000, context_length=2048,
                                   multi_group=0,
                                   )
          )

def estimate_vector(min=150, max=200, multi_group=16):
    for head_dim in [128]: #, 256, 384:
        for num_heads in range(48, 112+32, 8):

            if multi_group > 0:
                if num_heads % multi_group > 0: continue
            for num_layers in range(num_heads-62, num_heads+100, 1):
                num_params = calculate_num_parameters(num_heads=num_heads,
                                                  head_dim=head_dim,
                                                  num_layers=num_layers,
                                                  vocab_size=50000,
                                                  context_length=2048,
                                                  multi_group=multi_group,
                                                  )
                if num_params < max and num_params > min:
                    print("Vector Head Dim {} {} heads {} layers {}B multi group {}".format(head_dim, num_heads, num_layers,
                                                         num_params, multi_group),
                    )
                """
                Vector Head Dim 256 64 heads 64 layers 181.25348864B
                Vector Head Dim 128 96 heads 110 layers 172.285841408B
                """



if __name__ == "__main__":
    estimate_gpt3()
    estimate_bloom() # 173.4 B

    print("large size model")
    estimate_vector(min=150, max=240, multi_group=16)
    """
    Vector Head Dim 128 112 heads 80 layers 169.8734843776B
    Vector Head Dim 128 112 heads 90 layers 191.014393856B
    Vector Head Dim 256 64 heads 64 layers 181.25348864B
    Vector Head Dim 256 64 heads 70 layers 198.166052864B
    """

    print("large size model - num groups 32")
    estimate_vector(min=150, max=200, multi_group=32)
    """
    Vector Head Dim 256 64 heads 60 layers 178.032001024B multi group 32
    """

    print("medium size model")
    estimate_vector(min=64, max=220, multi_group=16)
    """
    Multi Group = 16
    Vector Head Dim 128 80 heads 68 layers 74.696235008B
    Vector Head Dim 256 48 heads 60 layers 97.284968448B multi group 16
    """
    print("------------")
    print("extra large model")
    estimate_vector(min=500, max=600, multi_group=16)
    """
    Multi Group = 32
    """