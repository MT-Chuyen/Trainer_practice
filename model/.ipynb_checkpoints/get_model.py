from model import FCN

def get_method(args):
    if args.method == 'FCN':
        model = FCN(args)
    return model