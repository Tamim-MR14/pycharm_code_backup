unit=int(input("enter the electricity unit:"))
if (unit<=50):
    bill=0.5*unit
    bill=bill+bill*0.2
elif (unit>50 and unit<=150):
    bill=25+(unit-50)*.75
    bill = bill + bill * 0.2
elif (unit>150 and unit<=250):
    bill=25+100*.75+(unit-150)*1.2
    bill = bill + bill * 0.2
else:
    bill=50+100*.75+100*1.2+(unit-250)*1.5
    bill = bill + bill * 0.2

print("The value is", end=' ',)
print(bill)