import numpy as np
import re
from test_model import decode_sequence, encoder_model, decoder_model, num_decoder_tokens, num_encoder_tokens, input_features_dict, target_features_dict, reverse_target_features_dict, max_decoder_seq_length, max_encoder_seq_length
from tensorflow.keras.models import load_model

model = load_model("training_model.h5")

class ChatBot:
    negative_commands = ('no', 'bad', 'terrible', 'annoying')
    exit_commands = ('end', 'break', 'bye', 'not anymore', 'exit')

    def start_chat(self):
        r = input("Are you looking for love?")
        while r not in self.negative_commands:
            if self.make_exit(r):
                print("Okay! Bye love!")
                break
            r  = input(self.generate_response(r))


    
    def make_exit(self, reply):
        for r in reply:
            if r in self.exit_commands:
                return True
    
    def string_to_matrix(self, input):
        tokens = re.findall(r"[\w']+|[^\s\w]", input)
        user_input_matrix = np.zeros((1, max_encoder_seq_length, num_encoder_tokens),dtype='float32')
        for timestep, token in enumerate(tokens):
            if token in input_features_dict:
                user_input_matrix[0, timestep, input_features_dict[token]] = 1.
        return user_input_matrix

    def generate_response(self, user_input):
        input_matrix = self.string_to_matrix(user_input)
        states_value = model.predict(input_matrix)
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        # Populate the first token of target sequence with the start token.
        target_seq[0, 0, target_features_dict['<START>']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        decoded_sentence = ''

        stop_condition = False
        while not stop_condition:
            # Run the decoder model to get possible 
            # output tokens (with probabilities) & states
            output_tokens, hidden_state, cell_state = decoder_model.predict(
            [target_seq] + states_value)

            # Choose token with highest probability
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = reverse_target_features_dict[sampled_token_index]
            decoded_sentence += " " + sampled_token

            # Exit condition: either hit max length
            # or find stop token.
            if (sampled_token == '<END>' or len(decoded_sentence) > max_decoder_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [hidden_state, cell_state]
        decoded_sentence.replace("<START>", '')
        decoded_sentence.replace("<END>", '')
        return decoded_sentence

bot = ChatBot()
bot.start_chat()

