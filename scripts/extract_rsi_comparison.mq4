#property copyright "RSI Comparison Extractor"
#property link      ""
#property version   "1.00"
#property strict
#property show_inputs

input int BarsToExport = 10000;
input string FileName = "rsi_comparison.csv";
input datetime StartDate = D'2024.01.01';
input int InitialDelay = 1;

void OnStart()
{
   MessageBox("This script will extract Royal Black RSI and standard RSI for comparison.\n\nMake sure you have scrolled back in the chart to load historical data.\n\nScript will start in " + IntegerToString(InitialDelay) + " seconds.", "RSI Extractor", MB_OK|MB_ICONINFORMATION);

   for(int i = InitialDelay; i > 0; i--)
   {
      Comment("Starting in ", i, " seconds...");
      Sleep(1000);
   }
   Comment("");

   int handle = FileOpen(FileName, FILE_WRITE|FILE_CSV|FILE_ANSI);

   if(handle != INVALID_HANDLE)
   {
      // Header
      FileWrite(handle, "Date", "Time", "RB_RSI", "RSI_9", "RSI_14", "RSI_21", "Close");

      int totalBars = iBars(Symbol(), PERIOD_H1);
      int processedBars = 0;

      Print("Starting extraction. Total bars available: ", totalBars);

      for(int i = totalBars - 1; i >= 0; i--)
      {
         datetime barTime = iTime(Symbol(), PERIOD_H1, i);
         if(barTime < StartDate) continue;

         // Professor's Royal Black RSI (buffer 0)
         double rbRSI = iCustom(Symbol(), PERIOD_H1, "Royal Black - RSI", 0, i);

         // Standard RSI for comparison
         double rsi9 = iRSI(Symbol(), PERIOD_H1, 9, PRICE_CLOSE, i);
         double rsi14 = iRSI(Symbol(), PERIOD_H1, 14, PRICE_CLOSE, i);
         double rsi21 = iRSI(Symbol(), PERIOD_H1, 21, PRICE_CLOSE, i);

         double close = iClose(Symbol(), PERIOD_H1, i);

         // Skip invalid values
         if(rbRSI == 0 || rbRSI == EMPTY_VALUE) continue;

         FileWrite(handle,
            TimeToStr(barTime, TIME_DATE),
            TimeToStr(barTime, TIME_MINUTES),
            DoubleToStr(rbRSI, 4),
            DoubleToStr(rsi9, 4),
            DoubleToStr(rsi14, 4),
            DoubleToStr(rsi21, 4),
            DoubleToStr(close, 5)
         );

         processedBars++;

         if(i % 500 == 0)
         {
            Comment("Processing... ", processedBars, " bars exported");
            if(IsStopped())
            {
               Print("Script stopped by user");
               FileClose(handle);
               return;
            }
         }
      }

      FileClose(handle);
      Comment("");
      Print("Export completed! ", processedBars, " bars exported to ", FileName);
      MessageBox("Export completed!\n\n" + IntegerToString(processedBars) + " bars exported to:\n" + FileName, "RSI Extractor", MB_OK|MB_ICONINFORMATION);
   }
   else
   {
      Print("Error opening file. Error: ", GetLastError());
      MessageBox("Error opening file!", "RSI Extractor", MB_OK|MB_ICONERROR);
   }
}
