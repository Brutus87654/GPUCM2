#---SAVE FORMAT---
#block_count
#nor_count
#led_count
#blocks_nor[nor_count]
#blocks_led[led_count]
#states[]
#led_poperties
#
#block={
#    u32 id
#    u8 input_count
#    u32 inputs[input_count]
#    f32 x
#    f32 y
#    f32 z
#    properties
#}
#
#properties:
#   nor: none
#   led:
#       u8 r
#       u8 g
#       u8 b
#       u8 opacity_on
#       u8 opacity_off
#

import numpy

class block():
    def __init__(self):
        self.block_type=0
        self.id=0
        self.input_count=0
        self.inputs=[]
        self.state=0
        self.x=0
        self.y=0
        self.z=0
        self.properties=[]

def sort_blocks(b):return [b.block_type,b.input_count,b.inputs,b.id,b.state]

def ceil(x):return (int(x)+1 if int(x)<x else int(x))

def clamp(x,minimum,maximum):return min(max(x,minimum),maximum)

input_filepath=input("enter input filepath:")
output_filepath=input("enter output filepath:")

with open(input_filepath,"r") as cm2_save:
    cm2_savestring=cm2_save.read()

cm2_savestring_blocks=list(filter(None,cm2_savestring.split('?')[0].split(';')))
if len(cm2_savestring.split('?'))>1:cm2_savestring_connections=list(filter(None,cm2_savestring.split('?')[1].split(';')))
else:cm2_savestring_connections=[]

blocks=[]
for block_id in range(len(cm2_savestring_blocks)):
    blocks.append(block())
    blocks[-1].block_type=int(cm2_savestring_blocks[block_id].split(',')[0])
    blocks[-1].id=block_id
    blocks[-1].state=(0 if cm2_savestring_blocks[block_id].split(',')[1]==''else int(cm2_savestring_blocks[block_id].split(',')[1]))
    blocks[-1].x=(0 if cm2_savestring_blocks[block_id].split(',')[2]==''else float(cm2_savestring_blocks[block_id].split(',')[2]))
    blocks[-1].y=(0 if cm2_savestring_blocks[block_id].split(',')[3]==''else float(cm2_savestring_blocks[block_id].split(',')[3]))
    blocks[-1].z=(0 if cm2_savestring_blocks[block_id].split(',')[4]==''else float(cm2_savestring_blocks[block_id].split(',')[4]))
    blocks[-1].properties=([] if cm2_savestring_blocks[block_id].split(',')[5]==''else cm2_savestring_blocks[block_id].split(',')[5].split('+'))

for connection in cm2_savestring_connections:
    blocks[int(connection.split(',')[1])-1].inputs.append(int(connection.split(',')[0])-1)
    blocks[int(connection.split(',')[1])-1].input_count+=1

blocks.sort(key=sort_blocks)

for b in blocks:
    new_inputs=[]
    for i in b.inputs:
        for b2 in range(len(blocks)):
            if blocks[b2].id==i:
                new_inputs.append(b2)
    b.inputs=new_inputs

for i in range(len(blocks)):
    blocks[i].id=i

states=[]
blocks_nor=[]
blocks_led=[]
for b in blocks:
    states.append(b.state)
    match (b.block_type):
        case 0:blocks_nor.append([b.id,b.input_count,b.inputs,b.x,b.y,b.z])
        case 6:blocks_led.append([b.id,b.input_count,b.inputs,b.x,b.y,b.z,(175 if b.properties==[] else int(b.properties[0])),
                                                                          (175 if b.properties==[] else int(b.properties[1])),
                                                                          (175 if b.properties==[] else int(b.properties[2])),
                                                                          (100 if b.properties==[] else int(b.properties[3])),
                                                                          (25  if b.properties==[] else int(b.properties[4]))])
        case _:print("invalid block type");exit()

states_compact=[]

states+=[0]*((32-len(states))%32)


for i in range(ceil(len(states)>>5)):
    states_compact+=[
        states[i*32+31]<<7|
        states[i*32+30]<<6|
        states[i*32+29]<<5|
        states[i*32+28]<<4|
        states[i*32+27]<<3|
        states[i*32+26]<<2|
        states[i*32+25]<<1|
        states[i*32+24],
        states[i*32+23]<<7|
        states[i*32+22]<<6|
        states[i*32+21]<<5|
        states[i*32+20]<<4|
        states[i*32+19]<<3|
        states[i*32+18]<<2|
        states[i*32+17]<<1|
        states[i*32+16],
        states[i*32+15]<<7|
        states[i*32+14]<<6|
        states[i*32+13]<<5|
        states[i*32+12]<<4|
        states[i*32+11]<<3|
        states[i*32+10]<<2|
        states[i*32+9]<<1|
        states[i*32+8],
        states[i*32+7]<<7|
        states[i*32+6]<<6|
        states[i*32+5]<<5|
        states[i*32+4]<<4|
        states[i*32+3]<<3|
        states[i*32+2]<<2|
        states[i*32+1]<<1|
        states[i*32]
    ]

with open(output_filepath,"wb") as savefile:
    save_list=[
        (len(blocks)>>24)    &255,
        (len(blocks)>>16)    &255,
        (len(blocks)>>8)     &255,
        (len(blocks))        &255,
        (len(blocks_nor)>>24)&255,
        (len(blocks_nor)>>16)&255,
        (len(blocks_nor)>>8) &255,
        (len(blocks_nor))    &255,
        (len(blocks_led)>>24)&255,
        (len(blocks_led)>>16)&255,
        (len(blocks_led)>>8) &255,
        (len(blocks_led))    &255,
    ]
    for b in blocks_nor:
        save_list+=[
            (b[0]>>24)&255,
            (b[0]>>16)&255,
            (b[0]>>8) &255,
            (b[0])    &255,
            (b[1])    &255,
        ]
        for i in b[2]:
            save_list+=[
                (i>>24)&255,
                (i>>16)&255,
                (i>>8) &255,
                (i)    &255,
            ]
        save_list+=list(bytes(numpy.float32(b[3])))
        save_list+=list(bytes(numpy.float32(b[4])))
        save_list+=list(bytes(numpy.float32(b[5])))
    for b in blocks_led:
        save_list+=[
            (b[0]>>24)&255,
            (b[0]>>16)&255,
            (b[0]>>8) &255,
            (b[0])    &255,
            (b[1])    &255,
        ]
        for i in b[2]:
            save_list+=[
                (i>>24)&255,
                (i>>16)&255,
                (i>>8) &255,
                (i)    &255,
            ]
        save_list+=list(bytes(numpy.float32(b[3])))
        save_list+=list(bytes(numpy.float32(b[4])))
        save_list+=list(bytes(numpy.float32(b[5])))
        save_list+=[clamp(b[6],0,255),clamp(b[7],0,255),clamp(b[8],0,255),clamp(b[9],0,100),clamp(b[10],0,100)]
    for s in states_compact:
        save_list+=[s]
    savefile.write(bytearray(save_list))
    print(save_list)