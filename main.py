# main.py
import argparse, os, glob, time
from trainer import Trainer
from image_loader import get_image_dataloaders

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='resnet50.a1_in1k', help="Backbone model name from timm")
    parser.add_argument('--num_classes', type=int, default=7, help="Number of emotion classes")
    parser.add_argument("--loadNumImages", type=int, default=-1, help="Max images to load per class. Use -1 to load all.")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--dataset_path', type=str, default='EmotionDataset')
    parser.add_argument('--savePath', type=str, default='checkpoints')
    parser.add_argument('--testInterval', type=int, default=1)
    parser.add_argument('--evaluation', action='store_true')
    parser.add_argument('--eval_model_path', type=str, default='path not specified')
    
    return parser.parse_args()


def main(args):
    trainLoader, valLoader = get_image_dataloaders(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        loadNumImages=args.loadNumImages,
        model_name=args.model_name
    )

    if args.evaluation:
        s = Trainer(args)
        if args.eval_model_path == "path not specified":
            print("Evaluation model parameters path has not been specified")
            return
        s.loadParameters(args.eval_model_path)
        print("Parameters loaded from path", args.eval_model_path)
        mAP = s.evaluate_network(loader=valLoader)
        print("mAP %2.2f%%" % (mAP))
        return

    args.modelSavePath = os.path.join(args.savePath, 'model')
    os.makedirs(args.modelSavePath, exist_ok=True)
    args.scoreSavePath = os.path.join(args.savePath, 'score.txt')

    modelfiles = glob.glob('%s/model_0*.model' % args.modelSavePath)
    modelfiles.sort()

    if len(modelfiles) >= 1:
        print("Model %s loaded from previous state!" % modelfiles[-1])
        epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
        s = Trainer(args, epoch=epoch)
        s.loadParameters(modelfiles[-1])
    else:
        epoch = 1
        s = Trainer(args, epoch=epoch)

    mAPs = []
    scoreFile = open(args.scoreSavePath, "a+")
    bestmAP = 0

    while epoch <= args.epochs:
        loss, lr = s.train_network(epoch=epoch, loader=trainLoader)
        
        if epoch % args.testInterval == 0:
            mAP = s.evaluate_network(epoch=epoch, loader=valLoader)
            mAPs.append(mAP)
            if mAP > bestmAP:
                bestmAP = mAP
                s.saveParameters(f"{args.modelSavePath}/best.model")
            print(time.strftime("%Y-%m-%d %H:%M:%S"), f"{epoch} epoch, mAP {mAP:.2f}%, bestmAP {bestmAP:.2f}%")
            scoreFile.write(f"{epoch} epoch, LR {lr}, LOSS {loss:.4f}, mAP {mAP:.2f}%, bestmAP {bestmAP:.2f}%\n")
            scoreFile.flush()

        epoch += 1


if __name__ == "__main__":
    args = get_args()
    main(args)
