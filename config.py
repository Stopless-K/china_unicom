
class Config():
    def __init__(self):
        self.data_path = r'data\\'

        self.drop_feature = ['gender', 'age', 'former_complaint_num','former_complaint_fee', 'user_id']
        self.label_name= 'current_service'
        self.out_path = r'out\\'

        self.idx_2_service = 'idx_2_service.pkl'