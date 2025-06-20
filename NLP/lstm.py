from manim import *
import random

def random_hex_color():
    return "#"+"".join(random.choices("0123456789ABCDEF", k=6))

class LSTMInputFlow(Scene):
    def construct(self):
        #camera color
        self.camera.background_color = WHITE

        #input draw
        x0 = Rectangle(width=5,height=1,color=BLUE)
        x0.shift(UP*3)
        x0.set_fill(BLUE,opacity=0.4)

        time_steps = VGroup()
        
        x0_label = MathTex(r"x_0",color=BLACK)
        x0_label.next_to(x0,LEFT)
        self.play(Create(x0))
        self.play(Write(x0_label))

        time_steps_C = []
        time_steps_r = []
        for i in range(7):
            t = Rectangle(width=0.5,height=0.25,color=PURPLE)
            t.set_fill(PURPLE,opacity=0.6)
            rotulo = MathTex(r't_{}'.format(i+1),color=BLACK,font_size=20)
            rotulo.move_to(t.get_center())
            group_t = VGroup([t,rotulo])
            time_steps.add(group_t)
            time_steps_C.append(Create(t))
            time_steps_r.append(Write(rotulo))

        time_steps.arrange(RIGHT,buff=0.1)
        time_steps.move_to(x0.get_center())

        
        self.play(LaggedStart(*time_steps_C,lag_ratio=0.1))
        self.play(LaggedStart(*time_steps_r,lag_ratio=0.1))            
        
        x0 = VGroup([x0,time_steps])
        x0 = VGroup([x0_label,x0])
        
        #LSTM draw
        lstm = Rectangle(width=3,height=2.5,color=BLACK)
        lstm.set_fill(BLACK,opacity=0.8)
        lstm_label = Text("LSTM",font_size=35,color=WHITE)
        lstm_label.move_to(lstm.get_center())
        self.play(Create(lstm))
        self.play(Write(lstm_label))
        lstm = VGroup([lstm,lstm_label])

        #input rotate
        moveto = x0.animate.next_to(lstm,LEFT*2).rotate(-PI/2)
        self.play(moveto)
        x0_label_r = x0[0].animate.rotate(PI/2)
        t_labels_r = [t.animate.rotate(PI/2) for t in x0[1][1]]
        self.play(LaggedStart(moveto,x0_label_r,*t_labels_r,lag_ratio=0.1))

        #draw H outputs
        H = Rectangle(width=5,height=1,color=RED)
        H.set_fill(RED,opacity=0.5)
        H_label = Text("h's",color=BLACK,font_size=30)
        H.next_to(lstm,RIGHT*2).rotate(-PI/2)
        self.play(Create(H))
        H_label.next_to(H,UP)
        self.play(Write(H_label))

        #pass each timestep along lstm
        
        current_height = 0
        buff = np.round((time_steps.height - time_steps[0].height*len(time_steps))/(len(time_steps)-1),5)
        padding_top = np.round((H.height - time_steps.height)/2,5)+time_steps[0].height/2
        for i,t in enumerate(time_steps):
            self.play(t.animate.move_to(lstm.get_center()),run_time=1)
            new_label = MathTex(r"h_{}".format(i+1),color=BLACK,font_size=20)
            self.play(LaggedStart(t.animate.set_color(ORANGE),Transform(t[1],new_label),lag_ratio=0.1))
            self.play(t.animate.move_to(H.get_top()+DOWN*(current_height+padding_top)))
            t.set_z_index(10)
            current_height += t.height + buff


        self.wait(2)
     


class LSTMInternalFlow(Scene):
    def construct(self):
        #camera color
        self.camera.background_color = WHITE

        #LSTM DRAW
        LSTMBox = Rectangle(width=5,height=3,color=RED)
        left_bottom_edge = LSTMBox.get_corner(DL)
        right_bottom_edge = LSTMBox.get_corner(DR)

        h_line = Line(start = left_bottom_edge+LEFT*0.5+UP*0.3,end=right_bottom_edge-RIGHT+UP*0.3,color=PURPLE)
        h_label = MathTex(r"H[t-1]",color=BLACK,font_size = 25)
        h_label.next_to(h_line,LEFT)
        


        
        x_line = Line(start= left_bottom_edge+DOWN*0.5+RIGHT*0.35,end=left_bottom_edge+UP*0.3+RIGHT*0.35,color=PURPLE)
        x_label = MathTex(r"X[t]",color=BLACK,font_size = 25)
        x_label.next_to(x_line,DOWN)

        left_top_edge = LSTMBox.get_corner(UL)
        right_top_edge = LSTMBox.get_corner(UR)
        c_line = Line(start = left_top_edge+LEFT*0.5+DOWN*0.3,end=right_top_edge+RIGHT*0.5+DOWN*0.3,color=BLUE)
        c_label = MathTex(r"C[t-1]",color=BLACK,font_size = 25)
        c_label.next_to(c_line,LEFT)
        c_label_end = MathTex(r"C[t]",color=BLACK,font_size = 25)
        c_label_end.next_to(c_line,RIGHT)

        f_line = Line(left_bottom_edge+UP*0.3+RIGHT*1,left_bottom_edge+UP*1+RIGHT*1,color=PURPLE)
        i_line = Line(left_bottom_edge+UP*0.3+RIGHT*2,left_bottom_edge+UP*1+RIGHT*2,color=PURPLE)
        cc_line = Line(left_bottom_edge+UP*0.3+RIGHT*3,left_bottom_edge+UP*1+RIGHT*3,color=PURPLE)
        o_line = Line(left_bottom_edge+UP*0.3+RIGHT*4,left_bottom_edge+UP*1+RIGHT*4,color=PURPLE)

        sigmoid = Rectangle(width=0.5,height=0.3,color=GREEN)
        sigmoid.set_fill(GREEN,opacity=0.5)
        sigmoid_label = MathTex(r"\sigma",color=BLACK,font_size=20)
        sigmoid_label.next_to(sigmoid,ORIGIN)
        sigmoid = VGroup([sigmoid,sigmoid_label])
        sigmoid.next_to(f_line,UP*0.1)

        sigmoid_i = sigmoid.copy()
        sigmoid_i.next_to(i_line,UP*0.1)

        sigmoid_o = sigmoid.copy()
        sigmoid_o.next_to(o_line,UP*0.1)

        tanh_l = MathTex(r"tanh",color=BLACK,font_size=20)
        tanh = sigmoid[0].copy()
        tanh_l.next_to(tanh,ORIGIN)
        tanh = VGroup([tanh,tanh_l])
        tanh.next_to(cc_line,UP*0.1)

        #operations
        dot_f = Circle(0.15,color=DARK_BLUE)
        dot_f.set_fill(DARK_BLUE,opacity=1)
        dot_inside = Circle(0.10,color=WHITE)
        dot_inside.move_to(dot_f.get_center())
        dot_m = Dot(dot_f.get_center(),color=WHITE,radius=0.025)
        dot_f = VGroup([dot_f,dot_inside,dot_m])
        dot_f.move_to(left_top_edge+DOWN*0.3+RIGHT)

        dot_i = dot_f.copy()
        dot_i.next_to(tanh,UP*2)

        dot_o = dot_i.copy()
        dot_o.next_to(sigmoid_o,RIGHT*0.8)
    
        plus_i = dot_f[0].copy()
        dot_inside_p = dot_inside.copy()
        plus_l = MathTex(r"+",color=WHITE,font_size=15)
        plus_l.move_to(plus_i.get_center())
        dot_inside_p.move_to(plus_i.get_center())
        plus_i = VGroup([plus_i,plus_l,dot_inside_p])
        plus_i.move_to(left_top_edge+DOWN*0.3+RIGHT*3)

        tanh_out = Ellipse(width=0.4,height=0.3,color=DARK_BLUE)
        tanh_out.set_fill(DARK_BLUE,opacity=1)
        tanh_o_l = tanh_l.copy().set_color(WHITE)
        tanh_o_l.font_size = 15
        tanh_o_l.move_to(tanh_out.get_center())
        tanh_out = VGroup([tanh_out,tanh_o_l])
        tanh_out.next_to(dot_o,UP*2.5)

        line_dot_f = Line(sigmoid.get_top(),dot_f.get_bottom(),color=PURPLE)

        line_dot_i = Line(sigmoid_i.get_top(),dot_i.get_center()-RIGHT,color=PURPLE)
        line_dot_i2 = Line(line_dot_i.get_top(),dot_i.get_left(),color=PURPLE)
        line_dot_i = VGroup([line_dot_i,line_dot_i2])
        line_c_dot_i = Line(plus_i.get_bottom(),dot_i.get_top(),color=PURPLE)
        line_tanh_dot_i = Line(tanh.get_top(),dot_i.get_bottom(),color=PURPLE)


        line_tanh_out = Line(tanh_out.get_top()+UP*0.45,tanh_out.get_top(),color=BLUE)
        line_tanh_out2 = Line(tanh_out.get_bottom(),dot_o.get_top(),color=BLUE)
        line_dot_o = Line(sigmoid_o.get_right(),dot_o.get_left(),color=PURPLE)



        line_h_out = Line(dot_o.get_bottom(),dot_o.get_bottom()+DOWN*0.73,color="#FF0000")
        line_h_out2 = Line(line_h_out.get_bottom(),right_bottom_edge+RIGHT*0.5+UP*0.3,color="#FF0000")
        line_h_out = VGroup([line_h_out,line_h_out2])
        
        h_label_end = MathTex(r"H[t]",color=BLACK,font_size = 25)
        h_label_end.next_to(line_h_out2,RIGHT)

        self.play(LaggedStart(Create(LSTMBox),Create(h_line),Create(x_line),Create(c_line),lag_ratio=0.1))
        self.play(LaggedStart(Write(h_label),Write(x_label),Write(h_label_end),Write(c_label),Write(c_label_end),lag_ratio=0.1))
        self.play(LaggedStart(Create(f_line),Create(i_line),Create(cc_line),Create(o_line),lag_ratio=0.1))
        self.play(LaggedStart(Create(sigmoid),Create(sigmoid_i),Create(tanh),Create(sigmoid_o),lag_ratio=0.1))
        self.play(LaggedStart(Create(dot_f),Create(dot_i),Create(dot_o),Create(plus_i),Create(tanh_out),lag_ratio=0.1))
        self.play(LaggedStart(Create(line_dot_f),Create(line_dot_i),
                    Create(line_c_dot_i),Create(line_tanh_dot_i),
                    Create(line_tanh_out),Create(line_tanh_out2),
                    Create(line_dot_o),Create(line_h_out),lag_ratio=0.1))




        # create matrix in and matrix out 
        
        input = VGroup()
        colors = {
                    i:color for i,color in 
                    enumerate(
                        [
                            '#cc6e8e', '#ec9e78', '#7653f5', '#58faf8',
                            '#a4b4c5', '#b158d9', '#c29a57'
                        ]
                    )
                }
        
        
        for i in range(7):
            time_steps = VGroup()
            color  = colors[i]
            for t in range(1,4):
                time_step = Square(side_length=0.5,color=color)
                time_step.set_fill(color,opacity=0.7)
                label = MathTex(r"\mathbf{{x_{{ {},t{} }} }}".format(i,t),font_size=15,color=BLACK)
                label.move_to(time_step.get_center())
                time_step = VGroup(time_step,label)
                time_steps.add(time_step)
            time_steps.arrange(RIGHT,buff=0.1)
            input.add(time_steps)
        
        input.arrange(DOWN,buff=0.1)
                
        input_mat = Rectangle(height=input.height*1.2,width=input.width*1.2,color=GRAY)
        input_mat.set_fill(GRAY,opacity=0.5)
        input.move_to(input_mat.get_center())
        label_i_mat = MathTex(r"X",font_size=20,color=BLACK)
        label_i_mat.next_to(input_mat,UP)
        input_mat = VGroup(input_mat,input,label_i_mat)
        input_mat.next_to(LSTMBox,LEFT*9)
        

        output_mat = input_mat[0].copy()
        output_mat.set_fill("#0e2f44",opacity=1)
        output_mat.set_color("#0e2f44")
        label_o_mat = MathTex(r"H",font_size=20,color=BLACK)
        label_o_mat.next_to(output_mat,UP)
        output_mat = VGroup(output_mat,label_o_mat)
        output_mat.next_to(LSTMBox,RIGHT*9)
        

        


        self.play(LaggedStart(Create(input_mat),Create(output_mat),lag_ratio=0.1))

        padding = (input_mat.width - input.width)/2
        buff = (input.width - time_steps[0].width*len(time_steps))/(len(time_steps)-1)
        padding_top = (input_mat.height - input.height)/3
        size_obj = time_steps[0].width
        H_matrix = VGroup()
        for t in range(0,3):
            time_step_samples = VGroup()
            moves = []
            H_curr = Square(side_length=0.5,color="#576C88")
            H_curr.set_fill("#576C88",opacity=1)
            batchs = VGroup()
            if t == 0:
                H_curr_text = r"\mathbf{0}"
                C_curr_text = r"\mathbf{0}"
            else:
                H_curr_text = r"\mathbf{{H_{}}}".format(t)
                C_curr_text = r"\mathbf{{C_{}}}".format(t)
            H_curr_label = MathTex(H_curr_text,font_size=15,color=BLACK)
            H_curr_label.move_to(H_curr.get_center())
            
            H_curr = VGroup(H_curr,H_curr_label)
            H_curr.move_to(h_line.get_left())
            for r in range(0,7):
                row = input_mat[1][r][t]
                time_step_samples.add(row)
                moves.append(row.animate.move_to(x_line.get_bottom()))

                row_h = row[0].copy()
                row_h_l = MathTex(r"\mathbf{{H_{{{},{}}}}}".format(r,t+1),font_size = 15,color=BLACK)
                row_h_l.move_to(row_h.get_center())
                row_h = VGroup(row_h,row_h_l)

                batchs.add(row_h)

            batchs.arrange(DOWN,buff=0.1)
            
            representation_x = Circle(radius=0.1,color="#00b4ff")
            representation_x.set_fill("#00b4ff",opacity=1)
            label_rep = MathTex(r"\mathbf{{t{}}}".format(t+1),font_size=10,color=BLACK)
            representation_x = VGroup(representation_x,label_rep)
            representation_x.move_to(x_line.get_bottom())

            representation_h = Circle(radius=0.1,color="#fdd3d5")
            representation_h.set_fill("#fdd3d5",opacity=1)
            label_rep_h = MathTex(H_curr_text,font_size=10,color=BLACK)
            representation_h = VGroup(representation_h,label_rep_h)
            representation_h.move_to(h_line.get_left())

            representation_c = Circle(radius=0.1,color="#fbc840")
            representation_c.set_fill("#fbc840",opacity=1)
            label_rep_c = MathTex(C_curr_text,font_size=10,color=BLACK)
            representation_c = VGroup(representation_c,label_rep_c)
            representation_c.move_to(c_line.get_left())


            self.play(LaggedStart(*moves,Create(H_curr),Create(representation_c),lag_ratio=0))
            self.play(
                LaggedStart(
                    Transform(time_step_samples,representation_x),
                    Transform(H_curr,representation_h),lag_ratio=0
                )
            )

            ## walking through LSTM
            
            #forget_gate

            self.play(
                LaggedStart(
                    time_step_samples.animate.move_to(x_line.get_top()),
                    H_curr.animate.move_to(x_line.get_top()),
                    representation_c.animate.move_to(dot_f.get_center()),
                    lag_ratio=0
                )
            )

            representations = VGroup(H_curr,time_step_samples)

            concat_hx = Rectangle(width=0.5,height=0.25,color="#72c097")
            concat_hx.set_fill("#72c097",opacity=1)
            label_concat = MathTex(r"\mathbf{{[H_{},Xt_{}]}}".format(t,t+1),font_size=10,color=BLACK)
            concat_hx = VGroup(concat_hx,label_concat)
            concat_hx.move_to(x_line.get_top())

            self.play(Transform(representations,concat_hx))
            
            input_flow = Circle(radius=0.2,color="#72c097")
            input_flow.set_fill("#72c097",opacity=1)
            input_flow_l = MathTex(r"\mathbf{{[H_{},Xt_{}]}}".format(t,t+1),font_size=10,color=BLACK)
            input_flow_l.move_to(input_flow.get_center())
            input_flow = VGroup(input_flow,input_flow_l)
            input_flow.move_to(x_line.get_top())

            self.play(Transform(representations,input_flow))

            self.play(representations.animate.move_to(f_line.get_bottom()))

            repre_F = representations.copy()
            self.play(repre_F.animate.move_to(sigmoid.get_center()))
            new_repre_F = Circle(radius=0.1,color="#7f7f7f")
            new_repre_F = new_repre_F.set_fill("#7f7f7f",opacity=1)
            new_repre_F_Legend = MathTex(r"\mathbf{{F_{}}}".format(t+1),font_size=10,color=BLACK)
            new_repre_F_Legend.move_to(new_repre_F.get_center())
            new_repre_F = VGroup(new_repre_F,new_repre_F_Legend)
            new_repre_F.move_to(sigmoid.get_center())

            self.play(Transform(repre_F,new_repre_F))
            self.play(repre_F.animate.move_to(dot_f.get_center()))
            representation_c_n = representation_c[0].copy()
            representation_c_n.set_fill("#fde9b2")
            representation_c_n.set_color("#fde9b2")
            representation_c[1].move_to(representation_c_n.get_center())
            representation_c_n = VGroup(representation_c_n,representation_c[1])
            self.play(LaggedStart(FadeOut(repre_F),Transform(representation_c,representation_c_n),lag_ratio=0.2))
            self.play(
                LaggedStart(
                    representations.animate.move_to(i_line.get_bottom()),
                    representation_c.animate.move_to(plus_i.get_center()),
                    lag_ratio=0.3
                )
            )
            repre_I = representations.copy()
            self.play(repre_I.animate.move_to(sigmoid_i.get_center()))
            new_repre_I = new_repre_F[0].copy()
            new_repre_I_leg = MathTex(r"\mathbf{{I_{}}}".format(t+1),font_size=10,color=BLACK)
            new_repre_I_leg.move_to(new_repre_I.get_center())
            new_repre_I = VGroup(new_repre_I,new_repre_I_leg)
            new_repre_I.move_to(sigmoid_i.get_center())
            self.play(Transform(repre_I,new_repre_I))
            self.play(repre_I.animate.move_to(line_dot_i2.get_left()))
            self.play(
                LaggedStart(
                    repre_I.animate.move_to(dot_i.get_center()),
                    representations.animate.move_to(cc_line.get_bottom())
                )
                
            )
            
            repre_tanh = representations.copy()
            self.play(repre_tanh.animate.move_to(tanh.get_center()))
            new_repre_tanh = new_repre_F[0].copy()
            new_repre_tanh.set_fill("#000000")
            new_repre_tanh.set_color("#000000")
            new_repre_tanh_l = MathTex(r"\mathbf{{\tilde{{C}}_{}}}".format(t+1),font_size=10,color=WHITE)
            new_repre_tanh_l.move_to(new_repre_tanh.get_center())
            new_repre_tanh = VGroup(new_repre_tanh,new_repre_tanh_l)
            new_repre_tanh.move_to(tanh.get_center())

            self.play(Transform(repre_tanh,new_repre_tanh))
            self.play(repre_tanh.animate.move_to(dot_i.get_center()))

            new_repre_tanh[0].set_fill("#515562")
            new_repre_tanh[0].set_color("#515562")
            new_repre_tanh[1].move_to(new_repre_tanh.get_center())
            new_repre_tanh.move_to(dot_i.get_center())

            self.play(
                LaggedStart(
                    FadeOut(repre_I),
                    Transform(repre_tanh,new_repre_tanh)
                )
            )
            self.play(repre_tanh.animate.move_to(plus_i.get_center()))
            
            representation_c_n2 = representation_c_n[0].copy()
            representation_c_n2.set_fill("#d6b588")
            representation_c_n2.set_color("#d6b588")
            representation_c[1].move_to(representation_c_n2.get_center())
            representation_c_n2 = VGroup(representation_c_n2,representation_c[1])
            representation_c_n2.move_to(plus_i.get_center())

            self.play(
                LaggedStart(
                    FadeOut(repre_tanh),
                    Transform(representation_c,representation_c_n2),
                    lag_ratio=0.1
                )
            )
            
            self.play(
                LaggedStart(
                    representation_c.animate.move_to(line_tanh_out.get_top()),
                    representations.animate.move_to(o_line.get_bottom()),
                    lag_ratio=0
                )
            )

            self.play(
                LaggedStart(
                    representation_c.animate.move_to(tanh_out.get_center()),
                    representations.animate.move_to(sigmoid_o.get_center()),
                    lag_ratio=0
                )
            )

            new_rep = new_repre_I[0].copy()
            new_rep_l = MathTex(r"\mathbf{{O_{}}}".format(t+1),font_size=10,color=WHITE)
            new_rep_l.move_to(new_rep.get_center())
            new_rep = VGroup(new_rep,new_rep_l)
            new_rep.move_to(sigmoid_o.get_center())


            new_rep_c = representation_c[0].copy()
            new_rep_c.set_fill(BLACK)
            new_rep_c.set_color(BLACK)

            representation_c[1].move_to(new_rep_c.get_center()).set_color(WHITE)
            new_rep_c = VGroup(new_rep_c,representation_c[1])

            
            self.play(
                LaggedStart(
                    Transform(representations,new_rep),
                    Transform(representation_c,new_rep_c),
                    lag_ratio=0
                )
                
            )

            self.play(
                LaggedStart(
                    representation_c.animate.move_to(dot_o.get_center()),
                    representations.animate.move_to(dot_o.get_center()),
                    lag_ratio=0
                )
            )

            output_h = representation_c[0].copy()
            output_h.set_fill("#464344")
            output_h.set_color("#464344")
            output_h_l = MathTex(r"\mathbf{{H_{}}}".format(t+1),font_size=10,color=WHITE)
            output_h_l.move_to(output_h.get_center())
            output_h = VGroup(output_h,output_h_l)

            self.play(
                LaggedStart(
                    FadeOut(representations),
                    Transform(representation_c,output_h),
                    lag_ratio=0.3
                )
            )
            
            self.play(
                representation_c.animate.move_to(line_h_out2.get_left())
            )
            self.play(
                representation_c.animate.move_to(line_h_out2.get_right())
            )

          
            
            location = output_mat.get_left()+RIGHT*(padding+((buff+size_obj)*t+0.25))-UP*(padding_top-size_obj/2)
            print(padding,size_obj,buff)
            self.play(representation_c.animate.move_to(location))

            batchs.move_to(location)

            self.play(Transform(representation_c,batchs))

            H_matrix.add(batchs)         
   

        self.wait(2)



def create_input_row(self, LSTM,reversed=False):
    
    x0 = Rectangle(width=1,height=2.5,color=BLUE)
    x0.set_fill(BLUE,opacity=0.4)
    x0_label = MathTex(r"x_0",color=BLACK,font_size=25)
    x0_label.next_to(x0,UP)
    x0 = VGroup(x0,x0_label)
    x0.next_to(LSTM,LEFT*2)
    self.play(Create(x0))
   
    time_steps = VGroup()
    
    if reversed:
        range_ = range(6,-1,-1)
    else:
        range_ = range(7)
    for i in range_:
        t = Rectangle(width=0.4,height=0.25,color=PURPLE)
        t.set_fill(PURPLE,opacity=0.6)
        rotulo = MathTex(r't_{}'.format(i+1),color=BLACK,font_size=15)
        rotulo.move_to(t.get_center())
        group_t = VGroup([t,rotulo])
        time_steps.add(group_t)

    time_steps.arrange(DOWN,buff=0.1)
    time_steps.move_to(x0[0].get_center())

    
    self.play(LaggedStart([Create(time_step) for time_step in time_steps],lag_ratio=0.1))
    
    x0 = VGroup([x0,time_steps])
    return x0

def LSTMdraw(self,shift_pos,color):
    lstm = Rectangle(width=2,height=1.5,color=color)
    lstm.set_fill(color,opacity=0.8)
    lstm_label = Text("LSTM",font_size=25,color=WHITE)
    lstm_label.move_to(lstm.get_center())
    lstm = VGroup([lstm,lstm_label])
    lstm.shift(shift_pos)
    self.play(Create(lstm))
    return lstm

def draw_h(self,lstm,color):
    #draw H outputs
    H = Rectangle(width=1,height=2.5,color=color)
    H.set_fill(color,opacity=0.5)
    H_label = MathTex(r"h's",color=BLACK,font_size=25)
    H_label.next_to(H,UP)
    H = VGroup(H,H_label)
    H.next_to(lstm,RIGHT*2)
    self.play(Create(H))
    return H

class BILSTMInputFlow(Scene):
    def construct(self):
        #camera color
        self.camera.background_color = WHITE

        
        #LSTM draw
        lstm1 = LSTMdraw(self,UP*2+LEFT*2.5,BLACK)
        
        #input draw
        x0 = create_input_row(self,lstm1)

        #LSTM draw2
        lstm2 = LSTMdraw(self,DOWN*2+LEFT*2.5,RED)


        biLSTM = Rectangle(width=2.75,height=6,color="#cccccc")
        biLSTM.set_fill("#cccccc",opacity=0.2)
        biLSTMLabel = Text("BiLSTM",color=BLACK,font_size=30)
        biLSTMLabel.move_to(biLSTM.get_center())

        biLSTM = VGroup(biLSTM,biLSTMLabel)
        biLSTM.shift(LEFT*2.5)
        self.play(Create(biLSTM))

        #reversed input draw 
        x0_reversed = create_input_row(self,lstm2,True)

        #h lstm1 
        h1 = draw_h(self,lstm1,"#383c40")
       
        #h lstm2
        h2 = draw_h(self,lstm2,"#922937")
    

        # #pass each timestep along lstm
        
        current_height = 0
        buff = np.round((x0[1].height - x0[1][0].height*len(x0[1]))/(len(x0[1])-1),5)
        padding_top = np.round((h1[0].height - x0[1].height)/2,5) + x0[1][0].height/2
        for i,t in enumerate(x0[1]):
            tr = x0_reversed[1][i]
            self.play(
                LaggedStart(
                    t.animate.move_to(lstm1.get_center()),
                    tr.animate.move_to(lstm2.get_center()),
                    lag_ratio=0
                ),
            run_time=1)
            new_label = MathTex(r"h_{{l,{}}}".format(i+1),color=BLACK,font_size=20)
            new_label.move_to(t[0].get_center())
            new_label_tr = MathTex(r"h_{{r,{}}}".format(len(x0[1])-i),color=BLACK,font_size=20)
            new_label_tr.move_to(tr[0].get_center())
            self.play(
                LaggedStart(
                    t.animate.set_color(ORANGE),Transform(t[1],new_label),
                    tr.animate.set_color("#094a85"),Transform(tr[1],new_label_tr)
                    ,lag_ratio=0
                )
            )
            self.play(
                LaggedStart(
                    t.animate.move_to(h1[0].get_top()+DOWN*(current_height+padding_top)),
                    tr.animate.move_to(h2[0].get_top()+DOWN*(current_height+padding_top)),
                    lag_ratio=0
                )
            )
            t.set_z_index(10)
            tr.set_z_index(10)
            current_height += t.height + buff

        bid_h = Rectangle(width=1.5,height=4.5,color="#7f2121")
        bid_h.set_fill("#7f2121",opacity=1)
        bid_h_l = Text("biH",color=BLACK,font_size=20)
        bid_h_l.next_to(bid_h,UP)
        bid_h = VGroup(bid_h,bid_h_l)
        bid_h.next_to(biLSTM,RIGHT*10)

        self.play(Create(bid_h))

        current_height = 0
        padding_top = 0.9
        buff = 0.35

        for i,t in enumerate(x0[1]):
            tr = x0_reversed[1][-(i+1)]
            
            biH = Rectangle(width=1,height=0.5,color = "#f4f3f3")
            biH.set_fill("#f4f3f3",opacity=1)
            biH.move_to(bid_h.get_top()+DOWN*(current_height+padding_top))
            self.play(Create(biH))
            
            self.play(
                LaggedStart(
                    t.animate.move_to(biH.get_center() + LEFT*0.25),
                    tr.animate.move_to(biH.get_center() + RIGHT*0.25),
                    lag_ratio=0
                )
            )

            current_height += t.height + buff

        self.wait(2)


        
     
