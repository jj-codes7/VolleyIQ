' VolleyIQ: double-click this file to start VolleyIQ. It opens in your browser.
'
' No console window appears. This script runs the project's own Python,
' .venv\Scripts\pythonw.exe, with volleyiq\launcher.py, which starts the web
' app in the background, waits until it answers, and opens the browser.
' All paths come from where this file is, so it also works through a
' shortcut. If something goes wrong, a message box says what.
'
' To stop VolleyIQ, click "Stop VolleyIQ" on the web page.
' VolleyIQ.bat does the same as this file, but with a console window.
'
' Advanced: options are passed to the launcher, e.g. --port 8766 --no-browser.

Option Explicit

Const EXIT_OK = 0
Const EXIT_REPORTED = 2   ' the launcher already showed a message (see volleyiq\launcher.py)
Const HIDDEN_WINDOW = 0

Dim fso, shell, root, pythonw, launcher, command, i, code
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

root = fso.GetParentFolderName(WScript.ScriptFullName)
pythonw = fso.BuildPath(root, ".venv\Scripts\pythonw.exe")
launcher = fso.BuildPath(root, "volleyiq\launcher.py")

If Not fso.FileExists(pythonw) Then
    Fail "Could not find the project's Python environment:" & vbCrLf & pythonw & vbCrLf & vbCrLf & _
         "See ""Setup"" in README.md, or double-click VolleyIQ.bat."
End If
If Not fso.FileExists(launcher) Then
    Fail "Could not find the VolleyIQ launcher:" & vbCrLf & launcher
End If

command = Quote(pythonw) & " " & Quote(launcher)
For i = 0 To WScript.Arguments.Count - 1
    command = command & " " & Quote(WScript.Arguments(i))
Next

shell.CurrentDirectory = root
On Error Resume Next
code = shell.Run(command, HIDDEN_WINDOW, True)   ' True: wait for the launcher's result
If Err.Number <> 0 Then
    Dim reason
    reason = Err.Description
    On Error GoTo 0
    Fail "Windows could not run the project's Python:" & vbCrLf & pythonw & vbCrLf & vbCrLf & reason & _
         vbCrLf & vbCrLf & "Double-click VolleyIQ.bat to see more."
End If
On Error GoTo 0

If code <> EXIT_OK And code <> EXIT_REPORTED Then
    Fail "VolleyIQ could not start: Python stopped with exit code " & code & "." & vbCrLf & vbCrLf & _
         "Double-click VolleyIQ.bat to see the full output."
End If
WScript.Quit code


Function Quote(text)
    Quote = Chr(34) & text & Chr(34)
End Function

Sub Fail(message)
    ' A message box when double-clicked (wscript); plain text when run with cscript, e.g. by the tests.
    If InStr(LCase(WScript.FullName), "cscript") > 0 Then
        WScript.Echo "VolleyIQ: " & message
    Else
        MsgBox message, vbCritical, "VolleyIQ"
    End If
    WScript.Quit 1
End Sub
