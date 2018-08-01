class MixedCrystalSignature:
    solid_thresh=0.5

    def __init__(self,solid_thresh):
        self.solid_thresh=solid_thresh
if __name__ == '__main__':
    MCS=MixedCrystalSignature(0.4)
    print('test1')