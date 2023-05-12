import tkinter as tk
from model import *


class CVD_GUI:
    def __init__(self):

        # Create the main window.
        self.main_window = tk.Tk()
        self.main_window.title("Sepsis Predictor")

        # Create two frames to group widgets.
        self.one_frame = tk.Frame()
        self.two_frame = tk.Frame()
        self.three_frame = tk.Frame()
        self.four_frame = tk.Frame()
        self.five_frame = tk.Frame()
        self.six_frame = tk.Frame()
        self.seven_frame = tk.Frame()
        self.eight_frame = tk.Frame()
        self.nine_frame = tk.Frame()
        self.ten_frame = tk.Frame()
        self.eleven_frame = tk.Frame()


        # Create the widgets for one frame. (title display)
        self.title_label = tk.Label(self.one_frame, text='SEPSIS PREDICTOR',fg="Blue", font=("Helvetica", 18))
        self.title_label.pack()



        # Create the widgets for three frame. (Plasma glucose input)
        self.PRG_label = tk.Label(self.three_frame, text='Plasma glucose:')
        self.PRG_entry = tk.Entry(self.three_frame, bg="white", fg="black", width=10)
        # self.Plasma glucose:_entry.insert(0,'50')
        self.PRG_label.pack(side='left')
        self.PRG_entry.pack(side='left')


        # Create the widgets for four frame. (Blood Work Result-1 (mu U/ml) input)
        self.PL_label = tk.Label(self.four_frame, text='Blood Work Result-1 (mu U/ml):')
        self.PL_entry = tk.Entry(self.four_frame, bg="white", fg="black", width=10)
        # self.Plasma glucose:_entry.insert(0,'50')
        self.PL_label.pack(side='left')
        self.PL_entry.pack(side='left')

        # Create the widgets for five frame. (PR (Blood Pressure (mm Hg))input)
        self.PR_label = tk.Label(self.five_frame, text='Blood Pressure (mm Hg):')
        self.PR_entry = tk.Entry(self.five_frame, bg="white", fg="black")
        #self.trestbp_entry.insert(0,'150')
        self.PR_label.pack(side='left')
        self.PR_entry.pack(side='left')


        # Create the widgets for six frame. (SK/ Blood Work Result-2 (mm) input)
        self.SK_label = tk.Label(self.six_frame, text='Blood Work Result-2 (mm):')
        self.SK_entry = tk.Entry(self.six_frame, bg="white", fg="black")
        #self.SK_entry.insert(0,250)
        self.SK_label.pack(side='left')
        self.SK_entry.pack(side='left')

        # Create the widgets for seven frame. (TS / Blood Work Result-3 (mu U/ml)  input)
        self.TS_label = tk.Label(self.seven_frame, text='Blood Work Result-3:')
        self.TS_entry = tk.Entry(self.seven_frame, bg="white", fg="black")
        # self.TS_entry.insert(0,250)
        self.TS_label.pack(side='left')
        self.TS_entry.pack(side='left')

        # Create the widgets for eight frame. (Body mass index (weight in kg/(height in m)^2)  input)
        self.M11_label = tk.Label(self.eight_frame, text='Body mass index (weight in kg/(height in m)^2):')
        self.M11_entry = tk.Entry(self.eight_frame, bg="white", fg="black")
        # self.TS_entry.insert(0,250)
        self.M11_label.pack(side='left')
        self.M11_entry.pack(side='left')

        # Create the widgets for nine frame. (BD2 -Blood Work Result-4 (mu U/ml) input)
        self.BD2_label = tk.Label(self.nine_frame, text='Blood Work Result-4:')
        self.BD2_entry = tk.Entry(self.nine_frame, bg="white", fg="black")
        #self.BD2_entry.insert(0,'150')
        self.BD2_label.pack(side='left')
        self.BD2_entry.pack(side='left')


        # Create the widgets for ten frame. (Age -patients age  (years)  input)
        self.Age_label = tk.Label(self.ten_frame, text='patients age :')
        self.Age_entry = tk.Entry(self.ten_frame, bg="white", fg="black")
        # self.oldpeak_entry.insert(0,'4.0')
        self.Age_label.pack(side='left')
        self.Age_entry.pack(side='left')



        #Create the widgets for fifteen frame = sepsis (prediction of sepsis)
        self.sepsis_predict_ta = tk.Text(self.eleven_frame,height = 10, width = 25,bg= 'light blue')

        #Create predict button and quit button
        self.btn_predict = tk.Button(self.eleven_frame, text='Predict Sepsis', command=self.predict_sepsis)
        self.btn_quit = tk.Button(self.eleven_frame, text='Quit', command=self.main_window.destroy)


        self.sepsis_predict_ta.pack(side='left')
        self.btn_predict.pack()
        self.btn_quit.pack()

        # Pack the frames.
        self.one_frame.pack()
        self.two_frame.pack()
        self.three_frame.pack()
        self.four_frame.pack()
        self.five_frame.pack()
        self.six_frame.pack()
        self.seven_frame.pack()
        self.eight_frame.pack()
        self.nine_frame.pack()
        self.ten_frame.pack()
        self.eleven_frame.pack()




        # Enter the tkinter main loop.
        tk.mainloop()

    def predict_sepsis(self):
        result_string = ""

        self.sepsis_predict_ta.delete(0.0, tk.END)
        plasma_glucose = self.PRG_entry.get()
        blood_work_result_1 = self.PL_entry.get()
        blood_pressure = self.PR_entry.get()
        blood_work_result_2 = self.SK_entry.get()
        blood_work_result_3 = self.TS_entry.get()
        body_mass_index = self.M11_entry.get()
        blood_work_result_4 = self.BD2_entry.get()
        patients_age = self.Age_entry.get()





        result_string += "===Patient Diagnosis=== \n"
        patient_info = (plasma_glucose,blood_work_result_1,blood_pressure, blood_work_result_2,\
                         blood_work_result_3,body_mass_index,\
                         blood_work_result_4, patients_age)


        sepsis_prediction =  best_model.predict([patient_info])
        disp_string = ("This prediction has an accuracy of:", str(round(model_accuracy,3)))

        result = sepsis_prediction

        if(sepsis_prediction == [0]):
            result_string = (disp_string, '\n', "0 - Patient has lower risk of developing Sepsis")
        else:
            result_string = (disp_string, '\n'+ "1 - Patient has higher risk of developing Sepsis")
        self.sepsis_predict_ta.insert('1.0',result_string)



my_cvd_GUI = CVD_GUI()

