class Logger:
    def __init__(self, log_path):
        self.log_path = log_path
        self.log_file = open(log_path, "w")
    
    def log(self, msg):
        self.log_file.write(msg + "\n")
        self.log_file.flush() 
    
    def args_log(self, args, args_part_msg=None):
        # args: argparse.Namespace, args_part_msg: {partition key: partition msg}
        if args_part_msg is None:
            for k, v in args.__dict__.items():
                self.log(f"{k}: {v}")
        else:
            for k, v in args.__dict__.items():
                if k in args_part_msg:
                    self.log(f"{args_part_msg[k]}")
                self.log(f"{k}: {v}")