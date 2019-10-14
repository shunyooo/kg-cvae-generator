#    Copyright (C) 2018 Ruotian Luo, Toyota Technological Institute at Chicago


import torch
import torch.nn

from .model_utils import dynamic_rnn


def inference_loop(cell, output_fn, embeddings,
                   encoder_state,
                   start_of_sequence_id, end_of_sequence_id,
                   maximum_length, num_decoder_symbols, context_vector,
                   decode_type='greedy'):

    """ A decoder function used during inference.
        In this decoder function we calculate the next input by applying an argmax across
        the feature dimension of the output from the decoder. This is a
        greedy-search approach. (Bahdanau et al., 2014) & (Sutskever et al., 2014)
        use beam-search instead.

      Args:
        output_fn: An output function to project your `cell_output` onto class
        logits.

        If `None` is supplied it will act as an identity function.

        encoder_state: The encoded state to initialize the decoder`.
        embeddings: The embeddings Module used for the decoder sized
        `[num_decoder_symbols, embedding_size]`.
        start_of_sequence_id: The start of sequence ID in the decoder embeddings.
        end_of_sequence_id: The end of sequence ID in the decoder embeddings.
        maximum_length: The maximum allowed of time steps to decode.
        num_decoder_symbols: The number of classes to decode at each time step.
        context_vector: an extra vector that should be appended to the input embedding

              done: A boolean vector to indicate which sentences has reached a
              `end_of_sequence_id`. This is used for early stopping by the
              `dynamic_rnn_decoder`. When `time>=maximum_length` a boolean vector with
              all elements as `true` is returned.

              context state: `context_state`, this decoder function does not
              modify the given context state. The context state could be modified when
              applying e.g. beam search.
    """
    batch_size = encoder_state.size(1)

    outputs = []
    context_state = []
    cell_state, cell_input, cell_output = encoder_state, None, None
    for time in range(maximum_length):
        if cell_output is None:
            # invariant that this is time == 0
            next_input_id = encoder_state.new_full((batch_size,), start_of_sequence_id, dtype=torch.long)
            # done: indicate which sentences reaches eos. used for early stopping
            done = encoder_state.new_zeros(batch_size, dtype=torch.uint8)
            cell_state = encoder_state
        else:
            cell_output = output_fn(cell_output)
            outputs.append(cell_output)

            if decode_type == 'sample':
                matrix_U = -1.0 * torch.log(
                    -1.0 * torch.log(cell_output.new_empty(cell_output.size()).uniform_()))
                next_input_id = torch.max(cell_output - matrix_U, 1)[1]
            elif decode_type == 'greedy':
                next_input_id = torch.max(cell_output, 1)[1]
            else:
                raise ValueError("unknown decode type")

            next_input_id = next_input_id * (~done).long()  # make sure the next_input_id to be 0 if done
            done = (next_input_id == end_of_sequence_id) | done
            # save the decoding results into context state
            context_state.append(next_input_id)

        next_input = embeddings(next_input_id)
        if context_vector is not None:
            next_input = torch.cat([next_input, context_vector], 1)
        if done.long().sum() == batch_size:
            break

        cell_output, cell_state = cell(next_input.unsqueeze(1), cell_state)
        # Squeeze the time dimension
        cell_output = cell_output.squeeze(1)

        # zero out done sequences
        cell_output = cell_output * (~done).float().unsqueeze(1)

    dec_out = torch.cat([_.unsqueeze(1) for _ in outputs], 1)
    pad_layer = torch.nn.ConstantPad1d((0, maximum_length - dec_out.shape[1] - 1), 0.0)
    dec_out = dec_out.permute(0, 2, 1)

    dec_out = pad_layer(dec_out)
    dec_out = dec_out.permute(0, 2, 1)
    dec_out[:, dec_out.shape[1]:maximum_length-1, 0] = 1.0

    out_result = dec_out, cell_state, torch.cat(
        [_.unsqueeze(1) for _ in context_state], 1)

    return out_result


def train_loop(cell, output_fn, inputs, init_state, context_vector, sequence_length, max_len):

    if context_vector is not None:
        inputs = torch.cat(
            [inputs, context_vector.unsqueeze(1).expand(inputs.size(0), inputs.size(1), context_vector.size(1))], 2)
    return dynamic_rnn(cell, inputs, sequence_length, max_len, init_state, output_fn) + (None,)
