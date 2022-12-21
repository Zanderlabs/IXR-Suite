import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
import tkinter.ttk as ttk

from z_flow import ZFlow
from z_flow.gui import ToolTip


class ZSuite():
    def __init__(self, root) -> None:
        # setting title
        root.title("Z-suite mini v0.2")
        # setting window size
        root.configure(width=400, height=600)
        root.configure(bg='#264653')
        root.resizable(False, False)

        # move window center
        winWidth = root.winfo_reqwidth()
        winwHeight = root.winfo_reqheight()
        posRight = int(root.winfo_screenwidth() / 2 - winWidth / 2)
        posDown = int(root.winfo_screenheight() / 2 - winwHeight / 2)
        root.geometry("+{}+{}".format(posRight, posDown))

        # title
        titleLabel = tk.Label(root)
        titleLabel["bg"] = "#264653"
        ft = tkfont.Font(family='Helvetica', size=25)
        titleLabel["font"] = ft
        titleLabel["fg"] = "#168aad"
        titleLabel["justify"] = "center"
        titleLabel["text"] = "Z-Suite Mini"
        titleLabel.place(x=0, y=20, width=400, height=25)

        # create input boxes
        self.create_boardid_input(root)
        self.create_timeout_input(root)
        self.create_reference_input(root)
        self.create_display_ref_input(root)
        self.create_calibration_input(root)
        self.create_power_history_input(root)
        self.create_scale_ent_input(root)
        self.create_center_ent_input(root)
        self.create_headstr_ent_input(root)

        # create connect button
        connectBtn = tk.Button(root, command=self.connectBtn_command)
        connectBtn["activebackground"] = "#989898"
        connectBtn["anchor"] = "w"
        connectBtn["bg"] = "#2a9d8f"
        ft = tkfont.Font(family='Helvetica', size=13)
        connectBtn["font"] = ft
        connectBtn["fg"] = "#000000"
        connectBtn["justify"] = "center"
        connectBtn["text"] = "  Connect"
        connectBtn.place(x=150, y=500, width=100, height=36)

    def create_boardid_input(self, root):
        self.boardidEnt = tk.Entry(root)
        self.boardidEnt["bg"] = "#e9f5db"
        self.boardidEnt["borderwidth"] = "1px"
        ft = tkfont.Font(family='Helvetica', size=13)
        self.boardidEnt["font"] = ft
        self.boardidEnt["fg"] = "#264653"
        self.boardidEnt["justify"] = "center"
        self.boardidEnt.insert(tk.END, '38')
        #self.boardidEnt["textvariable"] = self.text_boardid
        self.boardidEnt.place(x=250, y=80, width=70, height=25)

        boardidLabel = tk.Label(root)
        boardidLabel["bg"] = "#264653"
        ft = tkfont.Font(family='Helvetica', size=11)
        boardidLabel["font"] = ft
        boardidLabel["fg"] = "#e9f5db"
        boardidLabel["justify"] = "center"
        boardidLabel["text"] = "board ID"
        boardidLabel.place(x=40, y=80, width=200, height=25)
        self.CreateToolTip(boardidLabel, text='Muse 2 is 38, Muse S is 39 (default = 39)')

    def create_timeout_input(self, root):
        self.timeoutEnt = tk.Entry(root)
        self.timeoutEnt["bg"] = "#e9f5db"
        self.timeoutEnt["borderwidth"] = "1px"
        ft = tkfont.Font(family='Helvetica', size=13)
        self.timeoutEnt["font"] = ft
        self.timeoutEnt["fg"] = "#264653"
        self.timeoutEnt["justify"] = "center"
        self.timeoutEnt.insert(tk.END, '30')
        self.timeoutEnt.place(x=250, y=120, width=70, height=25)

        timeoutLabel = tk.Label(root)
        timeoutLabel["bg"] = "#264653"
        ft = tkfont.Font(family='Helvetica', size=12)
        timeoutLabel["font"] = ft
        timeoutLabel["fg"] = "#e9f5db"
        timeoutLabel["justify"] = "center"
        timeoutLabel["text"] = "timeout"
        timeoutLabel.place(x=40, y=120, width=200, height=25)
        self.CreateToolTip(timeoutLabel, text='Timeout when trying to connect to the board (default = 30)')

    def create_reference_input(self, root):
        self.reference_ent = tk.Entry(root)
        self.reference_ent["bg"] = "#e9f5db"
        self.reference_ent["borderwidth"] = "1px"
        ft = tkfont.Font(family='Helvetica', size=13)
        self.reference_ent["font"] = ft
        self.reference_ent["fg"] = "#264653"
        self.reference_ent["justify"] = "center"
        self.reference_ent.insert(tk.END, 'mean')
        self.reference_ent.place(x=250, y=160, width=70, height=25)

        reference_label = tk.Label(root)
        reference_label["bg"] = "#264653"
        ft = tkfont.Font(family='Helvetica', size=12)
        reference_label["font"] = ft
        reference_label["fg"] = "#e9f5db"
        reference_label["justify"] = "center"
        reference_label["text"] = "reference"
        reference_label.place(x=40, y=160, width=200, height=25)
        self.CreateToolTip(reference_label,
                           text="Determines what type of re-reference to use.\n"
                                " - none: No re-referencing is applied.\n"
                                " - mean (default): Use the mean of the four frontal and temporal electrodes.\n"
                                " - ref: Use the reference electrode(s) as a reference.")

    def create_display_ref_input(self, root):
        self.display_ref_ent = tk.Entry(root)
        self.display_ref_ent["bg"] = "#e9f5db"
        self.display_ref_ent["borderwidth"] = "1px"
        ft = tkfont.Font(family='Helvetica', size=13)
        self.display_ref_ent["font"] = ft
        self.display_ref_ent["fg"] = "#264653"
        self.display_ref_ent["justify"] = "center"
        self.display_ref_ent.insert(tk.END, '0')
        self.display_ref_ent.place(x=250, y=200, width=70, height=25)

        display_ref_label = tk.Label(root)
        display_ref_label["bg"] = "#264653"
        ft = tkfont.Font(family='Helvetica', size=12)
        display_ref_label["font"] = ft
        display_ref_label["fg"] = "#e9f5db"
        display_ref_label["justify"] = "center"
        display_ref_label["text"] = "display reference"
        display_ref_label.place(x=40, y=200, width=200, height=25)
        self.CreateToolTip(display_ref_label,
                           text="Set '1' to displays signal of the reference electrode(s) on the dashboard. ")

    def create_headstr_ent_input(self, root):
        self.headStrengthEnt = tk.Entry(root)
        self.headStrengthEnt["bg"] = "#e9f5db"
        self.headStrengthEnt["borderwidth"] = "1px"
        ft = tkfont.Font(family='Helvetica', size=13)
        self.headStrengthEnt["font"] = ft
        self.headStrengthEnt["fg"] = "#264653"
        self.headStrengthEnt["justify"] = "center"
        self.headStrengthEnt.insert(tk.END, '0.2')
        self.headStrengthEnt.place(x=250, y=400, width=70, height=25)

        headStrength = tk.Label(root)
        headStrength["bg"] = "#264653"
        ft = tkfont.Font(family='Helvetica', size=12)
        headStrength["font"] = ft
        headStrength["fg"] = "#e9f5db"
        headStrength["justify"] = "center"
        headStrength["text"] = "head impact strength"
        headStrength.place(x=40, y=400, width=200, height=25)
        self.CreateToolTip(
            headStrength, text='The amount of impact the head movement can have on the brain power (default = 0.2)')

    def create_center_ent_input(self, root):
        self.centerEnt = tk.Entry(root)
        self.centerEnt["bg"] = "#e9f5db"
        self.centerEnt["borderwidth"] = "1px"
        ft = tkfont.Font(family='Helvetica', size=13)
        self.centerEnt["font"] = ft
        self.centerEnt["fg"] = "#264653"
        self.centerEnt["justify"] = "center"
        self.centerEnt.insert(tk.END, '0.4')
        self.centerEnt.place(x=250, y=360, width=70, height=25)

        center = tk.Label(root)
        center["bg"] = "#264653"
        ft = tkfont.Font(family='Helvetica', size=12)
        center["font"] = ft
        center["fg"] = "#e9f5db"
        center["justify"] = "center"
        center["text"] = "center"
        center.place(x=40, y=360, width=200, height=25)
        self.CreateToolTip(center, text='The value around which the brainpower should be centered.\n'
                           'If at 0.5 then your "normal" brain power is 0.5.\n'
                           'If at 0.3 then your normal brain power is 0.3 and it is more difficult to reach max focus (default = 0.4).')

    def create_scale_ent_input(self, root):
        self.scaleEnt = tk.Entry(root)
        self.scaleEnt["bg"] = "#e9f5db"
        self.scaleEnt["borderwidth"] = "1px"
        ft = tkfont.Font(family='Helvetica', size=13)
        self.scaleEnt["font"] = ft
        self.scaleEnt["fg"] = "#264653"
        self.scaleEnt["justify"] = "center"
        self.scaleEnt.insert(tk.END, '1.0')
        self.scaleEnt.place(x=250, y=320, width=70, height=25)

        scale = tk.Label(root)
        scale["bg"] = "#264653"
        ft = tkfont.Font(family='Helvetica', size=12)
        scale["font"] = ft
        scale["fg"] = "#e9f5db"
        scale["justify"] = "center"
        scale["text"] = "scale"
        scale.place(x=40, y=320, width=200, height=25)
        self.CreateToolTip(scale, text='Adjusts the scale of the BCI.\n'
                           'SMALLER values make it EASIER to reach maximum and minimum \n'
                           '(recommended between 0.7 and 1.3, default = 1).')

    def create_power_history_input(self, root):
        self.powerHistoryEnt = tk.Entry(root)
        self.powerHistoryEnt["bg"] = "#e9f5db"
        self.powerHistoryEnt["borderwidth"] = "1px"
        ft = tkfont.Font(family='Helvetica', size=13)
        self.powerHistoryEnt["font"] = ft
        self.powerHistoryEnt["fg"] = "#264653"
        self.powerHistoryEnt["justify"] = "center"
        self.powerHistoryEnt.insert(tk.END, '10')
        self.powerHistoryEnt.place(x=250, y=280, width=70, height=25)

        powerHistoryLabel = tk.Label(root)
        powerHistoryLabel["bg"] = "#264653"
        ft = tkfont.Font(family='Helvetica', size=12)
        powerHistoryLabel["font"] = ft
        powerHistoryLabel["fg"] = "#e9f5db"
        powerHistoryLabel["justify"] = "center"
        powerHistoryLabel["text"] = "power history length"
        powerHistoryLabel.place(x=40, y=280, width=200, height=25)
        self.CreateToolTip(powerHistoryLabel,
                           text='Duration of the current brain power measurement in s (default = 10)')

    def create_calibration_input(self, root):
        self.calibrationHistoryEnt = tk.Entry(root)
        self.calibrationHistoryEnt["bg"] = "#e9f5db"
        self.calibrationHistoryEnt["borderwidth"] = "1px"
        ft = tkfont.Font(family='Helvetica', size=13)
        self.calibrationHistoryEnt["font"] = ft
        self.calibrationHistoryEnt["fg"] = "#264653"
        self.calibrationHistoryEnt["justify"] = "center"
        self.calibrationHistoryEnt.insert(tk.END, '600')
        self.calibrationHistoryEnt.place(x=250, y=240, width=70, height=25)

        calibrationHistoryLabel = tk.Label(root)
        calibrationHistoryLabel["bg"] = "#264653"
        ft = tkfont.Font(family='Helvetica', size=12)
        calibrationHistoryLabel["font"] = ft
        calibrationHistoryLabel["fg"] = "#e9f5db"
        calibrationHistoryLabel["justify"] = "center"
        calibrationHistoryLabel["text"] = "calibration history length"
        calibrationHistoryLabel.place(x=40, y=240, width=200, height=25)
        self.CreateToolTip(calibrationHistoryLabel,
                           text='Duration of the rolling calibration in seconds (default = 600)')

    def connectBtn_command(self) -> None:
        arguments = [
            '--board-id', str(self.boardidEnt.get()),
            '--timeout', str(self.timeoutEnt.get()),
            '--reference', str(self.reference_ent.get()),
            '--calib-length', str(self.calibrationHistoryEnt.get()),
            '--power-length', str(self.powerHistoryEnt.get()),
            '--scale', str(self.scaleEnt.get()),
            '--offset', str(self.centerEnt.get()),
            '--head-impact', str(self.headStrengthEnt.get()),
        ]
        if self.display_ref_ent.get() == '1':
            arguments.append('--display-ref')

        zflow = ZFlow(args=arguments)
        x = threading.Thread(target=zflow.run, daemon=True)
        x.start()

    @staticmethod
    def CreateToolTip(widget: tk.Label, text: str) -> None:
        toolTip = ToolTip(widget)

        def enter(event: str) -> None:
            toolTip.showtip(text)

        def leave(event: str) -> None:
            toolTip.hidetip()

        widget.bind('<Enter>', enter)
        widget.bind('<Leave>', leave)
