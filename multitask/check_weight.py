import torch 

model_1_state_dict = torch.load('/user/asessa/tesi/multitask/AGE_LORA_PE/ckpt/mtl_PE-Core-B16-224_ul1_1.pt')['backbone_state_dict']
model_2_state_dict = torch.load('/user/asessa/tesi/multitask/AGE_LORA_PE/ckpt/mtl_PE-Core-B16-224_ul1_2.pt')['backbone_state_dict']





def compare_state_dicts(dict1, dict2):
    if len(dict1) != len(dict2):
        print("Models have a different number of layers.")
        return False

    for (key1, val1), (key2, val2) in zip(dict1.items(), dict2.items()):
        if key1 != key2:
            print(f"Layer names do not match: {key1} vs {key2}")
            return False
        if not torch.equal(val1, val2):
            print(f"Weights of layer '{key1}' are not equal.")
            return False

    print("The weights of the two models are equal.")
    return True

# Compare the loaded state dictionaries
are_equal = compare_state_dicts(model_1_state_dict, model_2_state_dict)


def find_differing_layers(dict1, dict2):
    differing_layers = []
    if len(dict1) != len(dict2):
        print("Models have a different number of layers.")
        # You might want to handle this case based on your needs,
        # for instance by finding the common layers and comparing them.
        return

    for (key1, val1), (key2, val2) in zip(dict1.items(), dict2.items()):
        if key1 != key2:
            print(f"Layer names do not match: {key1} vs {key2}")
            continue # Move to the next layer
        if not torch.equal(val1, val2):
            differing_layers.append(key1)
        else:
            print(f"Weights of layer '{key1}' are equal.")

    if differing_layers:
        print("The following layers have different weights:")
        for layer in differing_layers:
            print(layer)
    else:
        print("All corresponding layers have the same weights.")

# Find and print the differing layers
find_differing_layers(model_1_state_dict, model_2_state_dict)
