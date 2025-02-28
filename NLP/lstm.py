from manim import *

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


        self.wait(2)