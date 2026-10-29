class StreamingDetokenizer:
    def __init__(self):
        self.text = ""
        self.offset = 0

    def add_token(self, piece):
        self.text += piece

    @property
    def last_segment(self):
        segment = self.text[self.offset:]
        self.offset = len(self.text)
        return segment


d = StreamingDetokenizer()
for piece in ["A", " large", " language"]:
    d.add_token(piece)
    if d.last_segment:
        print(d.last_segment, end="", flush=True)
print("\n<<<原样结束>>>")

d2 = StreamingDetokenizer()
for piece in ["A", " large", " language"]:
    d2.add_token(piece)
    seg = d2.last_segment
    if seg:
        print(seg, end="", flush=True)
print("\n<<<单次读取结束>>>")

d3 = StreamingDetokenizer()
d3.add_token("A")
print("连读三次:", repr(d3.last_segment), repr(d3.last_segment), repr(d3.last_segment))
