import sys
sys.path.append("../scop_classification_by_PRoBERTa")

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def parse_tensorboard(path, img_name=None, img_format="png"):
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    # make sure the scalars are in the event accumulator tags
    # print(ea.Tags()["scalars"]) # shows the tags
    
    labels = ["Train loss (best possible=0.0)", "Val loss (best possible=0.0)", "Val accuracy (best possible=1.0)"]
    for j, tag in enumerate(["train loss", "val loss", "acc"]):
        x, y=[],[]
        for i, event in enumerate(ea.Scalars(tag)):
            # print(event.step, event.value)
            x.append(event.step)
            y.append(event.value)
            # plt.plot(event.step, event.value)
            # () 
            if i==200: break
        plt.plot(x, y, label=labels[j])
    plt.legend()
    plt.xlabel("Number of epochs")
    plt.ylabel("Cross-entropy loss/Accuracy")
    if img_name==None: plt.show()
    else: plt.savefig(f"outputs/images/{img_name}.{img_format}", dpi=300, format=img_format, bbox_inches='tight', pad_inches=0.0)

parse_tensorboard(path="outputs/tensorboard_runs/CW_1e-05_64_300_cuda/events.out.tfevents.1652220562.node056.orc.gmu.edu.8499.0",
                  img_name="PRoBERTa_train_val_loss_accuracy_vs_epochs",
                  img_format="png")   