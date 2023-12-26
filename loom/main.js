const { app, BrowserWindow, ipcMain, dialog, Menu, MenuItem } = require('electron');
const fs = require('fs');

let mainWindow;

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 800,
        height: 600,
        webPreferences: {
            nodeIntegration: true,
	    contextIsolation: false,
        }
    });



  // Get the existing menu template
  const existingMenuTemplate = Menu.getApplicationMenu().items.map(item => {
    return {
      label: item.label,
      submenu: item.submenu.items,
    };
  });

  // Define new items for the File menu
  const fileMenuItems = [
    {
      label: 'Save',
      accelerator: 'CmdOrCtrl+S',
      click() {
        mainWindow.webContents.send('invoke-action', 'save-file');
      }
    },
    {
      label: 'Load',
      accelerator: 'CmdOrCtrl+O',
      click() {
        mainWindow.webContents.send('invoke-action', 'load-file');
      }
    },
    { type: 'separator' },  // Separator
  ];

  // Find the File menu in the existing template
  const fileMenuIndex = existingMenuTemplate.findIndex(item => item.label === 'File');

  if (fileMenuIndex >= 0) {
    // If File menu exists, append new items to it
    existingMenuTemplate[fileMenuIndex].submenu = fileMenuItems.concat(existingMenuTemplate[fileMenuIndex].submenu);
  } else {
    // If File menu doesn't exist, add it
    existingMenuTemplate.unshift({
      label: 'File',
      submenu: fileMenuItems
    });
  }

  // Build and set the new menu
  const newMenu = Menu.buildFromTemplate(existingMenuTemplate);
  Menu.setApplicationMenu(newMenu);
    
    mainWindow.loadFile('index.html');

    mainWindow.on('closed', function () {
        mainWindow = null;
    });
}

let autoSavePath = null;

ipcMain.handle('save-file', async (event, data) => {
  let filePath;
  if (autoSavePath) {
    filePath = autoSavePath;
  } else {
    const { filePath: chosenPath } = await dialog.showSaveDialog(mainWindow, {
      title: 'Save File',
      filters: [{ name: 'JSON Files', extensions: ['json'] }],
    });
    filePath = chosenPath;
    autoSavePath = chosenPath;  // Update auto-save path
  }

  if (filePath) {
    fs.writeFileSync(filePath, JSON.stringify(data));
  }
});

ipcMain.handle('load-file', async (event) => {
  const { filePaths } = await dialog.showOpenDialog(mainWindow, {
    title: 'Load File',
    filters: [{ name: 'JSON Files', extensions: ['json'] }],
    properties: ['openFile'],
  });

  if (filePaths && filePaths.length > 0) {
    const content = fs.readFileSync(filePaths[0], 'utf8');
    autoSavePath = filePaths[0];  // Update auto-save path
    return JSON.parse(content);
  }
});

ipcMain.handle('auto-save', (event, data) => {
  if (autoSavePath) {
    fs.writeFileSync(autoSavePath, JSON.stringify(data));
  }
});


app.whenReady().then(createWindow);

app.on('window-all-closed', function () {
    if (process.platform !== 'darwin') app.quit();
});

app.on('activate', function () {
    if (mainWindow === null) createWindow();
});

ipcMain.on('show-context-menu', (event) => {
  const contextMenu = Menu.buildFromTemplate([
    { label: 'Cut', role: 'cut' },
    { label: 'Copy', role: 'copy' },
    { label: 'Paste', role: 'paste' },
    { type: 'separator' },
    { label: 'Select All', role: 'selectAll' },
  ]);

  contextMenu.popup(BrowserWindow.fromWebContents(event.sender));
});
