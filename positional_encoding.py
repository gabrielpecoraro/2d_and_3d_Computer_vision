import torch


def positional_encoding_all(in_tensor, num_frequencies, max_freq_exp, min_freq_exp):
    # Scale the input from [0, 2*pi]
    scaled_in_tensor = 2 * torch.pi * in_tensor

    # Generate the frequency spectrum
    freqs = 2 ** torch.linspace(
        min_freq_exp, max_freq_exp, num_frequencies, device=in_tensor.device()
    )

    # Generate encoded inputs
    scaled_inputs = scaled_in_tensor.unsqueeze(-1) * freqs
    encoded_inputs = torch.cat(
        [torch.sin(scaled_inputs), torch.cos(scaled_inputs)], dim=-1
    )

    return encoded_inputs.view(in_tensor[:-1], -1)


def positional_encoding_two(nums_frequencies, x):
    frequencies = [2**i for i in range(nums_frequencies)]
    encoding = []

    for freq in frequencies:
        encoding.append(torch.sin(freq * x))
        encoding.append(torch.cos(freq * x))

    # Concatenate tensor accross the last dimension
    return torch.cat(encoding, dim=-1)
