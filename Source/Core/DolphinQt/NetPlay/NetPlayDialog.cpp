// Copyright 2017 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "DolphinQt/NetPlay/NetPlayDialog.h"

#include <QAction>
#include <QActionGroup>
#include <QApplication>
#include <QClipboard>
#include <QCheckBox>
#include <QComboBox>
#include <QFileDialog>
#include <QGridLayout>
#include <QGroupBox>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QSplitter>
#include <QTableWidget>
#include <QTextBrowser>
#include <QDialogButtonBox>

#include <algorithm>
#include <sstream>

#include "Common/CommonPaths.h"
#include "Common/Config/Config.h"
#include "Common/HttpRequest.h"
#include "Common/Logging/Log.h"
#include "Common/TraversalClient.h"

#include "Core/Boot/Boot.h"
#include "Core/Config/GraphicsSettings.h"
#include "Core/Config/MainSettings.h"
#include "Core/Config/NetplaySettings.h"
#include "Core/ConfigManager.h"
#include "Core/Core.h"
#ifdef HAS_LIBMGBA
#include "Core/HW/GBACore.h"
#endif
#include "Core/IOS/FS/FileSystem.h"
#include "Core/NetPlayServer.h"
#include "Core/SyncIdentifier.h"

#include "DolphinQt/NetPlay/ChunkedProgressDialog.h"
#include "DolphinQt/NetPlay/GameListDialog.h"
#include "DolphinQt/NetPlay/MD5Dialog.h"
#include "DolphinQt/NetPlay/PadMappingDialog.h"
#include "DolphinQt/QtUtils/ModalMessageBox.h"
#include "DolphinQt/QtUtils/QueueOnObject.h"
#include "DolphinQt/QtUtils/RunOnObject.h"
#include "DolphinQt/Resources.h"
#include "DolphinQt/Settings.h"
#include "DolphinQt/Settings/GameCubePane.h"

#include "UICommon/DiscordPresence.h"
#include "UICommon/GameFile.h"
#include "UICommon/UICommon.h"

#include "VideoCommon/NetPlayChatUI.h"
#include "VideoCommon/NetPlayGolfUI.h"
#include "VideoCommon/RenderBase.h"
#include "VideoCommon/VideoConfig.h"
#include <Common/CitrusRequest.cpp>
#include "Common/FileUtil.h"

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#endif
#include <Core/StateAuxillary.h>

namespace
{
QString InetAddressToString(const TraversalInetAddress& addr)
{
  QString ip;

  if (addr.isIPV6)
  {
    ip = QStringLiteral("IPv6-Not-Implemented");
  }
  else
  {
    const auto ipv4 = reinterpret_cast<const u8*>(addr.address);
    ip = QString::number(ipv4[0]);
    for (u32 i = 1; i != 4; ++i)
    {
      ip += QStringLiteral(".");
      ip += QString::number(ipv4[i]);
    }
  }

  return QStringLiteral("%1:%2").arg(ip, QString::number(ntohs(addr.port)));
}
}  // namespace

NetPlayDialog::NetPlayDialog(const GameListModel& game_list_model,
                             StartGameCallback start_game_callback, QWidget* parent)
    : QDialog(parent), m_game_list_model(game_list_model),
      m_start_game_callback(std::move(start_game_callback))
{
  setWindowFlags(windowFlags() & ~Qt::WindowContextHelpButtonHint);

  setWindowTitle(tr("NetPlay"));
  setWindowIcon(Resources::GetAppIcon());

  m_pad_mapping = new PadMappingDialog(this);
  m_md5_dialog = new MD5Dialog(this);
  m_chunked_progress_dialog = new ChunkedProgressDialog(this);

  ResetExternalIP();
  CreateChatLayout();
  CreatePlayersLayout();
  CreateMainLayout();
  LoadSettings();
  ConnectWidgets();

  auto& settings = Settings::Instance().GetQSettings();

  restoreGeometry(settings.value(QStringLiteral("netplaydialog/geometry")).toByteArray());
  m_splitter->restoreState(settings.value(QStringLiteral("netplaydialog/splitter")).toByteArray());

  srand(time(0));
}

NetPlayDialog::~NetPlayDialog()
{
  auto& settings = Settings::Instance().GetQSettings();

  settings.setValue(QStringLiteral("netplaydialog/geometry"), saveGeometry());
  settings.setValue(QStringLiteral("netplaydialog/splitter"), m_splitter->saveState());
}

void NetPlayDialog::CreateMainLayout()
{
  m_main_layout = new QGridLayout;
  m_game_button = new QPushButton;
  m_start_button = new QPushButton(tr("Start"));
  m_buffer_size_box = new QSpinBox;
  m_buffer_label = new QLabel(tr("Buffer:"));
  m_quit_button = new QPushButton(tr("Quit"));
  m_splitter = new QSplitter(Qt::Horizontal);
  m_menu_bar = new QMenuBar(this);
  m_ranked_box = new QCheckBox(tr("Ranked"));
  m_ranked_box->setToolTip(
      tr("Enabling Ranked Mode requires you to still submit your games to the ranked bot,\n"
         "Currently, it only marks games in our database as ranked."));

  m_coin_flipper = new QPushButton(tr("Coin flip"));
  m_data_menu = m_menu_bar->addMenu(tr("Data"));
  m_data_menu->setToolTipsVisible(true);
  m_write_save_data_action = m_data_menu->addAction(tr("Write Save Data"));
  m_write_save_data_action->setCheckable(true);
  m_load_wii_action = m_data_menu->addAction(tr("Load Wii Save"));
  m_load_wii_action->setCheckable(true);
  m_sync_save_data_action = m_data_menu->addAction(tr("Sync Saves"));
  m_sync_save_data_action->setCheckable(true);
  // m_sync_codes_action = m_data_menu->addAction(tr("Sync AR/Gecko Codes"));
  // m_sync_codes_action->setCheckable(true);
  m_sync_all_wii_saves_action = m_data_menu->addAction(tr("Sync All Wii Saves"));
  m_sync_all_wii_saves_action->setCheckable(true);
  m_strict_settings_sync_action = m_data_menu->addAction(tr("Strict Settings Sync"));
  m_strict_settings_sync_action->setToolTip(
      tr("This will sync additional graphics settings, and force everyone to the same internal "
         "resolution.\nMay prevent desync in some games that use EFB reads. Please ensure everyone "
         "uses the same video backend."));
  m_strict_settings_sync_action->setCheckable(true);

  m_network_menu = m_menu_bar->addMenu(tr("Network"));
  m_network_menu->setToolTipsVisible(true);
  m_fixed_delay_action = m_network_menu->addAction(tr("Fair Input Delay"));
  m_fixed_delay_action->setToolTip(
      tr("Each player sends their own inputs to the game, with equal buffer size for all players, "
         "configured by the host.\nSuitable for competitive games where fairness and minimal "
         "latency are most important."));
  m_fixed_delay_action->setCheckable(true);
  //m_host_input_authority_action = m_network_menu->addAction(tr("Host Input Authority"));
  //m_host_input_authority_action->setToolTip(
  //    tr("Host has control of sending all inputs to the game, as received from other players, "
  //       "giving the host zero latency but increasing latency for others.\nSuitable for casual "
  //       "games with 3+ players, possibly on unstable or high latency connections."));
  //m_host_input_authority_action->setCheckable(true);
  //m_golf_mode_action = m_network_menu->addAction(tr("Golf Mode"));
  //m_golf_mode_action->setToolTip(
  //    tr("Identical to Host Input Authority, except the \"Host\" (who has zero latency) can be "
  //       "switched at any time.\nSuitable for turn-based games with timing-sensitive controls, "
  //       "such as golf."));
  //m_golf_mode_action->setCheckable(true);

  m_network_mode_group = new QActionGroup(this);
  m_network_mode_group->setExclusive(true);
  m_network_mode_group->addAction(m_fixed_delay_action);
  //m_network_mode_group->addAction(m_host_input_authority_action);
  //m_network_mode_group->addAction(m_golf_mode_action);
  m_fixed_delay_action->setChecked(true);
  m_ranked_box->setChecked(false);

  m_md5_menu = m_menu_bar->addMenu(tr("Checksum"));
  m_md5_menu->addAction(tr("Current game"), this, [this] {
    Settings::Instance().GetNetPlayServer()->ComputeMD5(m_current_game_identifier);
  });
  m_md5_menu->addAction(tr("Other game..."), this, [this] {
    GameListDialog gld(m_game_list_model, this);

    if (gld.exec() != QDialog::Accepted)
      return;
    Settings::Instance().GetNetPlayServer()->ComputeMD5(gld.GetSelectedGame().GetSyncIdentifier());
  });
  m_md5_menu->addAction(tr("SD Card"), this, [] {
    Settings::Instance().GetNetPlayServer()->ComputeMD5(
        NetPlay::NetPlayClient::GetSDCardIdentifier());
  });

  m_other_menu = m_menu_bar->addMenu(tr("Other"));
  m_record_input_action = m_other_menu->addAction(tr("Record Inputs"));
  m_record_input_action->setCheckable(true);
  m_golf_mode_overlay_action = m_other_menu->addAction(tr("Show Golf Mode Overlay"));
  m_golf_mode_overlay_action->setCheckable(true);
  m_hide_remote_gbas_action = m_other_menu->addAction(tr("Hide Remote GBAs"));
  m_hide_remote_gbas_action->setCheckable(true);

  m_game_button->setDefault(false);
  m_game_button->setAutoDefault(false);

  m_sync_save_data_action->setChecked(true);
  // m_sync_codes_action->setChecked(true);

  m_main_layout->setMenuBar(m_menu_bar);

  m_main_layout->addWidget(m_game_button, 0, 0, 1, -1);
  m_main_layout->addWidget(m_splitter, 1, 0, 1, -1);

  m_splitter->addWidget(m_chat_box);
  m_splitter->addWidget(m_players_box);

  auto* options_widget = new QGridLayout;

  options_widget->addWidget(m_start_button, 0, 0, Qt::AlignVCenter);
  options_widget->addWidget(m_buffer_label, 0, 1, Qt::AlignVCenter);
  options_widget->addWidget(m_buffer_size_box, 0, 2, Qt::AlignVCenter);
  options_widget->addWidget(m_quit_button, 0, 5, Qt::AlignVCenter | Qt::AlignRight);
  options_widget->setColumnStretch(5, 1000);
  options_widget->addWidget(m_ranked_box, 0, 3, Qt::AlignVCenter);
  options_widget->addWidget(m_coin_flipper, 0, 4, Qt::AlignVCenter);

  m_main_layout->addLayout(options_widget, 2, 0, 1, -1, Qt::AlignRight);
  m_main_layout->setRowStretch(1, 1000);

  setLayout(m_main_layout);
}

void NetPlayDialog::CreateChatLayout()
{
  m_chat_box = new QGroupBox(tr("Chat"));
  m_chat_edit = new QTextBrowser;
  m_chat_type_edit = new QLineEdit;
  m_chat_send_button = new QPushButton(tr("Send"));

  // This button will get re-enabled when something gets entered into the chat box
  m_chat_send_button->setEnabled(false);
  m_chat_send_button->setDefault(false);
  m_chat_send_button->setAutoDefault(false);

  m_chat_edit->setReadOnly(true);

  auto* layout = new QGridLayout;

  layout->addWidget(m_chat_edit, 0, 0, 1, -1);
  layout->addWidget(m_chat_type_edit, 1, 0);
  layout->addWidget(m_chat_send_button, 1, 1);

  m_chat_box->setLayout(layout);
}

void NetPlayDialog::CreatePlayersLayout()
{
  m_players_box = new QGroupBox(tr("Players"));
  m_room_box = new QComboBox;
  m_hostcode_label = new QLabel;
  m_hostcode_action_button = new QPushButton(tr("Copy"));
  m_players_list = new QTableWidget;
  m_kick_button = new QPushButton(tr("Kick Player"));
  m_assign_ports_button = new QPushButton(tr("Assign Controller Ports"));

  m_players_list->setTabKeyNavigation(false);
  m_players_list->setColumnCount(5);
  m_players_list->verticalHeader()->hide();
  m_players_list->setSelectionBehavior(QAbstractItemView::SelectRows);
  m_players_list->horizontalHeader()->setStretchLastSection(true);
  m_players_list->horizontalHeader()->setHighlightSections(false);

  for (int i = 0; i < 4; i++)
    m_players_list->horizontalHeader()->setSectionResizeMode(i, QHeaderView::ResizeToContents);

  auto* layout = new QGridLayout;

  layout->addWidget(m_room_box, 0, 0);
  layout->addWidget(m_hostcode_label, 0, 1);
  layout->addWidget(m_hostcode_action_button, 0, 2);
  layout->addWidget(m_players_list, 1, 0, 1, -1);
  layout->addWidget(m_kick_button, 2, 0, 1, -1);
  layout->addWidget(m_assign_ports_button, 3, 0, 1, -1);

  m_players_box->setLayout(layout);
}

void NetPlayDialog::ConnectWidgets()
{
  // Players
  connect(m_room_box, qOverload<int>(&QComboBox::currentIndexChanged), this,
          &NetPlayDialog::UpdateGUI);
  connect(m_hostcode_action_button, &QPushButton::clicked, [this] {
    if (m_is_copy_button_retry)
      g_TraversalClient->ReconnectToServer();
    else
      QApplication::clipboard()->setText(m_hostcode_label->text());
  });
  connect(m_players_list, &QTableWidget::itemSelectionChanged, [this] {
    int row = m_players_list->currentRow();
    m_kick_button->setEnabled(row > 0 &&
                              !m_players_list->currentItem()->data(Qt::UserRole).isNull());
  });
  connect(m_kick_button, &QPushButton::clicked, [this] {
    auto id = m_players_list->currentItem()->data(Qt::UserRole).toInt();
    Settings::Instance().GetNetPlayServer()->KickPlayer(id);
  });
  connect(m_assign_ports_button, &QPushButton::clicked, [this] {
    m_pad_mapping->exec();

    Settings::Instance().GetNetPlayServer()->SetPadMapping(m_pad_mapping->GetGCPadArray());
    Settings::Instance().GetNetPlayServer()->SetGBAConfig(m_pad_mapping->GetGBAArray(), true);
    Settings::Instance().GetNetPlayServer()->SetWiimoteMapping(m_pad_mapping->GetWiimoteArray());
  });

  // Chat
  connect(m_chat_send_button, &QPushButton::clicked, this, &NetPlayDialog::OnChat);
  connect(m_chat_type_edit, &QLineEdit::returnPressed, this, &NetPlayDialog::OnChat);
  connect(m_chat_type_edit, &QLineEdit::textChanged, this,
          [this] { m_chat_send_button->setEnabled(!m_chat_type_edit->text().isEmpty()); });

  // Other
  connect(m_buffer_size_box, qOverload<int>(&QSpinBox::valueChanged), [this](int value) {
    if (value == m_buffer_size)
      return;
    INFO_LOG_FMT(NETPLAY, "State changed signal for Buffer");
    auto client = Settings::Instance().GetNetPlayClient();
    auto server = Settings::Instance().GetNetPlayServer();
    if (server && !m_host_input_authority)
      server->AdjustPadBufferSize(value);
    else
      client->AdjustPadBufferSize(value);
  });

  connect(m_ranked_box, &QCheckBox::stateChanged, [this](bool is_ranked) {
    m_current_ranked_value = is_ranked;
    StateAuxillary::setIsRanked(is_ranked);
    if (is_ranked)
    {
      INFO_LOG_FMT(NETPLAY, "Ranked is enabled");
      DisplayMessage(tr("Ranked Mode Enabled"), "mediumseagreen");
    }
    else
    {
      INFO_LOG_FMT(NETPLAY, "Ranked is disabled");
      DisplayMessage(tr("Ranked Mode Disabled"), "crimson");
    }
    Settings::Instance().GetNetPlayClient()->SendRankedState(is_ranked);
    /*

    auto client = Settings::Instance().GetNetPlayClient();
    auto server = Settings::Instance().GetNetPlayServer();
    INFO_LOG_FMT(NETPLAY, "State changed signal for Ranked Box");
    if (server)
      server->AdjustRankedBox(is_ranked);
    else
      client->AdjustRankedBox(is_ranked);
    */
  });

  connect(m_coin_flipper, &QPushButton::clicked, this, &NetPlayDialog::OnCoinFlip);

  const auto hia_function = [this](bool enable) {
    if (m_host_input_authority != enable)
    {
      auto server = Settings::Instance().GetNetPlayServer();
      if (server)
        server->SetHostInputAuthority(enable);
    }
  };

  /*connect(m_host_input_authority_action, &QAction::toggled, this,
          [hia_function] { hia_function(true); });*/
  //connect(m_golf_mode_action, &QAction::toggled, this, [hia_function] { hia_function(true); });
  connect(m_fixed_delay_action, &QAction::toggled, this, [hia_function] { hia_function(false); });

  connect(m_start_button, &QPushButton::clicked, this, &NetPlayDialog::OnStart);
  connect(m_quit_button, &QPushButton::clicked, this, &NetPlayDialog::reject);

  connect(m_game_button, &QPushButton::clicked, [this] {
    GameListDialog gld(m_game_list_model, this);
    if (gld.exec() == QDialog::Accepted)
    {
      Settings& settings = Settings::Instance();

      const UICommon::GameFile& game = gld.GetSelectedGame();
      const std::string netplay_name = m_game_list_model.GetNetPlayName(game);

      settings.GetNetPlayServer()->ChangeGame(game.GetSyncIdentifier(), netplay_name);
      Settings::GetQSettings().setValue(QStringLiteral("netplay/hostgame"),
                                        QString::fromStdString(netplay_name));
    }
  });

  connect(&Settings::Instance(), &Settings::EmulationStateChanged, this, [=](Core::State state) {
    if (isVisible())
    {
      GameStatusChanged(state != Core::State::Uninitialized);
      if ((state == Core::State::Uninitialized || state == Core::State::Stopping) &&
          !m_got_stop_request)
      {
        Settings::Instance().GetNetPlayClient()->RequestStopGame();
      }
      if (state == Core::State::Uninitialized)
        DisplayMessage(tr("Stopped game"), "red");
    }
  });

  connect(m_sync_save_data_action, &QAction::toggled, this,
          [this](bool checked) { m_sync_all_wii_saves_action->setEnabled(checked); });

  // SaveSettings() - Save Hosting-Dialog Settings

  connect(m_buffer_size_box, qOverload<int>(&QSpinBox::valueChanged), this,
          &NetPlayDialog::SaveSettings);
  connect(m_write_save_data_action, &QAction::toggled, this, &NetPlayDialog::SaveSettings);
  connect(m_load_wii_action, &QAction::toggled, this, &NetPlayDialog::SaveSettings);
  connect(m_sync_save_data_action, &QAction::toggled, this, &NetPlayDialog::SaveSettings);
  // connect(m_sync_codes_action, &QAction::toggled, this, &NetPlayDialog::SaveSettings);
  connect(m_record_input_action, &QAction::toggled, this, &NetPlayDialog::SaveSettings);
  connect(m_strict_settings_sync_action, &QAction::toggled, this, &NetPlayDialog::SaveSettings);
  // connect(m_host_input_authority_action, &QAction::toggled, this, &NetPlayDialog::SaveSettings);
  connect(m_sync_all_wii_saves_action, &QAction::toggled, this, &NetPlayDialog::SaveSettings);
  // connect(m_golf_mode_action, &QAction::toggled, this, &NetPlayDialog::SaveSettings);
  connect(m_golf_mode_overlay_action, &QAction::toggled, this, &NetPlayDialog::SaveSettings);
  connect(m_fixed_delay_action, &QAction::toggled, this, &NetPlayDialog::SaveSettings);
  connect(m_hide_remote_gbas_action, &QAction::toggled, this, &NetPlayDialog::SaveSettings);
}

void NetPlayDialog::SendMessage(const std::string& msg)
{
  Settings::Instance().GetNetPlayClient()->SendChatMessage(msg);

  DisplayMessage(
      QStringLiteral("%1: %2").arg(QString::fromStdString(m_nickname), QString::fromStdString(msg)),
      "");
}

void NetPlayDialog::OnChat()
{
  QueueOnObject(this, [this] {
    auto msg = m_chat_type_edit->text().toStdString();

    if (msg.empty())
      return;

    m_chat_type_edit->clear();

    SendMessage(msg);
  });
}

void NetPlayDialog::OnCoinFlip()
{
  if (!IsHosting())
    return;

  int flip = rand() % 2;
  Settings::Instance().GetNetPlayClient()->SendCoinFlip(flip);
}

void NetPlayDialog::OnCoinFlipResult(int coinVal)
{
  if (coinVal == 1)
  {
    DisplayMessage(tr("Heads"), "darkorange");
  }
  else
  {
    DisplayMessage(tr("Tails"), "darkblue");
  }
}

void NetPlayDialog::OnRankedChanged(bool is_ranked)
{
  QueueOnObject(this, [this, is_ranked] {
    const QSignalBlocker blocker(m_ranked_box);
    m_ranked_box->setChecked(is_ranked);
  });

  m_current_ranked_value = is_ranked;
  StateAuxillary::setIsRanked(is_ranked);

  if (is_ranked)
  {
    INFO_LOG_FMT(NETPLAY, "Ranked is enabled");
    DisplayMessage(tr("Ranked Mode Enabled"), "mediumseagreen");
  }
  else
  {
    INFO_LOG_FMT(NETPLAY, "Ranked is disabled");
    DisplayMessage(tr("Ranked Mode Disabled"), "crimson");
  }
}

void NetPlayDialog::OnIndexAdded(bool success, const std::string error)
{
  DisplayMessage(success ? tr("Successfully added to the NetPlay index") :
                           tr("Failed to add this session to the NetPlay index: %1")
                               .arg(QString::fromStdString(error)),
                 success ? "green" : "red");
}

void NetPlayDialog::OnIndexRefreshFailed(const std::string error)
{
  DisplayMessage(QString::fromStdString(error), "red");
}

void NetPlayDialog::OnStart()
{
  if (!Settings::Instance().GetNetPlayClient()->DoAllPlayersHaveGame())
  {
    if (ModalMessageBox::question(
            this, tr("Warning"),
            tr("Not all players have the game. Do you really want to start?")) == QMessageBox::No)
      return;
  }

  if (m_strict_settings_sync_action->isChecked() && Config::Get(Config::GFX_EFB_SCALE) == 0)
  {
    ModalMessageBox::critical(
        this, tr("Error"),
        tr("Auto internal resolution is not allowed in strict sync mode, as it depends on window "
           "size.\n\nPlease select a specific internal resolution."));
    return;
  }

  const auto game = FindGameFile(m_current_game_identifier);
  if (!game)
  {
    PanicAlertFmtT("Selected game doesn't exist in game list!");
    return;
  }

  if (Settings::Instance().GetNetPlayServer()->RequestStartGame())
  {
    SetOptionsEnabled(false);
    DisplayActiveGeckoCodes();
  }
}

void NetPlayDialog::reject()
{
  if (ModalMessageBox::question(this, tr("Confirmation"),
                                tr("Are you sure you want to quit NetPlay?")) == QMessageBox::Yes)
  {
    QDialog::reject();
  }
}

void NetPlayDialog::show(std::string nickname, bool use_traversal)
{
  m_nickname = nickname;
  m_use_traversal = use_traversal;
  m_buffer_size = 0;
  m_old_player_count = 0;

  m_room_box->clear();
  m_chat_edit->clear();
  m_chat_type_edit->clear();

  bool is_hosting = Settings::Instance().GetNetPlayServer() != nullptr;

  if (is_hosting)
  {
    if (use_traversal)
      m_room_box->addItem(tr("Room ID"));
    m_room_box->addItem(tr("External"));

    for (const auto& iface : Settings::Instance().GetNetPlayServer()->GetInterfaceSet())
    {
      const auto interface = QString::fromStdString(iface);
      m_room_box->addItem(iface == "!local!" ? tr("Local") : interface, interface);
    }
  }

  m_data_menu->menuAction()->setVisible(is_hosting);
  m_network_menu->menuAction()->setVisible(is_hosting);
  m_md5_menu->menuAction()->setVisible(is_hosting);
#ifdef HAS_LIBMGBA
  m_hide_remote_gbas_action->setVisible(is_hosting);
#else
  m_hide_remote_gbas_action->setVisible(false);
#endif
  m_start_button->setHidden(!is_hosting);
  m_kick_button->setHidden(!is_hosting);
  m_assign_ports_button->setHidden(!is_hosting);
  m_room_box->setHidden(!is_hosting);
  m_hostcode_label->setHidden(!is_hosting);
  m_hostcode_action_button->setHidden(!is_hosting);
  m_game_button->setEnabled(is_hosting);
  m_kick_button->setEnabled(false);
  m_coin_flipper->setEnabled(true);

  SetOptionsEnabled(true);

  QDialog::show();
  UpdateGUI();
}

void NetPlayDialog::ResetExternalIP()
{
  m_external_ip_address = Common::Lazy<std::string>([]() -> std::string {
    Common::HttpRequest request;
    // ENet does not support IPv6, so IPv4 has to be used
    request.UseIPv4();
    Common::HttpRequest::Response response =
        request.Get("https://ip.dolphin-emu.org/", {{"X-Is-Dolphin", "1"}});

    if (response.has_value())
      return std::string(response->begin(), response->end());
    return "";
  });
}

void NetPlayDialog::UpdateDiscordPresence()
{
#ifdef USE_DISCORD_PRESENCE
  // both m_current_game and m_player_count need to be set for the status to be displayed correctly
  if (m_player_count == 0 || m_current_game_name.empty())
    return;

  const auto use_default = [this]() {
    Discord::UpdateDiscordPresence(m_player_count, Discord::SecretType::Empty, "",
                                   m_current_game_name);
  };

  if (Core::IsRunning())
    return use_default();

  if (IsHosting())
  {
    if (g_TraversalClient)
    {
      const auto host_id = g_TraversalClient->GetHostID();
      if (host_id[0] == '\0')
        return use_default();

      Discord::UpdateDiscordPresence(m_player_count, Discord::SecretType::RoomID,
                                     std::string(host_id.begin(), host_id.end()),
                                     m_current_game_name);
    }
    else
    {
      if (m_external_ip_address->empty())
        return use_default();
      const int port = Settings::Instance().GetNetPlayServer()->GetPort();

      Discord::UpdateDiscordPresence(
          m_player_count, Discord::SecretType::IPAddress,
          Discord::CreateSecretFromIPAddress(*m_external_ip_address, port), m_current_game_name);
    }
  }
  else
  {
    use_default();
  }
#endif
}

void NetPlayDialog::UpdateGUI()
{
  auto client = Settings::Instance().GetNetPlayClient();
  auto server = Settings::Instance().GetNetPlayServer();
  if (!client)
    return;

  // Update Player List
  const auto players = client->GetPlayers();

  if (static_cast<int>(players.size()) != m_player_count && m_player_count != 0)
    QApplication::alert(this);

  m_player_count = static_cast<int>(players.size());

  int selection_pid = m_players_list->currentItem() ?
                          m_players_list->currentItem()->data(Qt::UserRole).toInt() :
                          -1;

  m_players_list->clear();
  m_players_list->setHorizontalHeaderLabels(
      {tr("Player"), tr("Game Status"), tr("Ping"), tr("Mapping"), tr("Revision")});
  m_players_list->setRowCount(m_player_count);

  static const std::map<NetPlay::SyncIdentifierComparison, QString> player_status{
      {NetPlay::SyncIdentifierComparison::SameGame, tr("OK")},
      {NetPlay::SyncIdentifierComparison::DifferentVersion, tr("Wrong Version")},
      {NetPlay::SyncIdentifierComparison::DifferentGame, tr("Not Found")},
  };

  for (int i = 0; i < m_player_count; i++)
  {
    const auto* p = players[i];

    auto* name_item = new QTableWidgetItem(QString::fromStdString(p->name));
    auto* status_item = new QTableWidgetItem(player_status.count(p->game_status) ?
                                                 player_status.at(p->game_status) :
                                                 QStringLiteral("?"));
    auto* ping_item = new QTableWidgetItem(QStringLiteral("%1 ms").arg(p->ping));
    auto* mapping_item =
        new QTableWidgetItem(QString::fromStdString(NetPlay::GetPlayerMappingString(
            p->pid, client->GetPadMapping(), client->GetGBAConfig(), client->GetWiimoteMapping())));
    auto* revision_item = new QTableWidgetItem(QString::fromStdString(p->revision));

    for (auto* item : {name_item, status_item, ping_item, mapping_item, revision_item})
    {
      item->setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable);
      item->setData(Qt::UserRole, static_cast<int>(p->pid));
    }

    m_players_list->setItem(i, 0, name_item);
    m_players_list->setItem(i, 1, status_item);
    m_players_list->setItem(i, 2, ping_item);
    m_players_list->setItem(i, 3, mapping_item);
    m_players_list->setItem(i, 4, revision_item);

    if (p->pid == selection_pid)
      m_players_list->selectRow(i);
  }

  if (m_old_player_count != m_player_count)
  {
    UpdateDiscordPresence();
    m_old_player_count = m_player_count;
  }

  if (!server)
    return;

  const bool is_local_ip_selected = m_room_box->currentIndex() > (m_use_traversal ? 1 : 0);
  if (is_local_ip_selected)
  {
    m_hostcode_label->setText(QString::fromStdString(
        server->GetInterfaceHost(m_room_box->currentData().toString().toStdString())));
    m_hostcode_action_button->setEnabled(true);
    m_hostcode_action_button->setText(tr("Copy"));
    m_is_copy_button_retry = false;
  }
  else if (m_use_traversal)
  {
    switch (g_TraversalClient->GetState())
    {
    case TraversalClient::State::Connecting:
      m_hostcode_label->setText(tr("Connecting"));
      m_hostcode_action_button->setEnabled(false);
      m_hostcode_action_button->setText(tr("..."));
      break;
    case TraversalClient::State::Connected:
    {
      if (m_room_box->currentIndex() == 0)
      {
        // Display Room ID.
        const auto host_id = g_TraversalClient->GetHostID();
        m_hostcode_label->setText(
            QString::fromStdString(std::string(host_id.begin(), host_id.end())));
      }
      else
      {
        // Externally mapped IP and port are known when using the traversal server.
        m_hostcode_label->setText(InetAddressToString(g_TraversalClient->GetExternalAddress()));
      }

      m_hostcode_action_button->setEnabled(true);
      m_hostcode_action_button->setText(tr("Copy"));
      m_is_copy_button_retry = false;
      break;
    }
    case TraversalClient::State::Failure:
      m_hostcode_label->setText(tr("Error"));
      m_hostcode_action_button->setText(tr("Retry"));
      m_hostcode_action_button->setEnabled(true);
      m_is_copy_button_retry = true;
      break;
    }
  }
  else
  {
    // Display External IP.
    if (!m_external_ip_address->empty())
    {
      const int port = Settings::Instance().GetNetPlayServer()->GetPort();
      m_hostcode_label->setText(QStringLiteral("%1:%2").arg(
          QString::fromStdString(*m_external_ip_address), QString::number(port)));
      m_hostcode_action_button->setEnabled(true);
    }
    else
    {
      m_hostcode_label->setText(tr("Unknown"));
      m_hostcode_action_button->setEnabled(false);
    }

    m_hostcode_action_button->setText(tr("Copy"));
    m_is_copy_button_retry = false;
  }
}

// NetPlayUI methods

void NetPlayDialog::BootGame(const std::string& filename,
                             std::unique_ptr<BootSessionData> boot_session_data)
{
  m_got_stop_request = false;
  m_start_game_callback(filename, std::move(boot_session_data));
}

void NetPlayDialog::StopGame()
{
  if (m_got_stop_request)
    return;

  m_got_stop_request = true;
  emit Stop();
}

bool NetPlayDialog::IsHosting() const
{
  return Settings::Instance().GetNetPlayServer() != nullptr;
}

void NetPlayDialog::Update()
{
  QueueOnObject(this, &NetPlayDialog::UpdateGUI);
}

void NetPlayDialog::DisplayMessage(const QString& msg, const std::string& color, int duration)
{
  QueueOnObject(m_chat_edit, [this, color, msg] {
    m_chat_edit->append(QStringLiteral("<font color='%1'>%2</font>")
                            .arg(QString::fromStdString(color), msg.toHtmlEscaped()));
  });

  QColor c(color.empty() ? QStringLiteral("white") : QString::fromStdString(color));

  if (g_ActiveConfig.bShowNetPlayMessages && Core::IsRunning())
    g_netplay_chat_ui->AppendChat(msg.toStdString(),
                                  {static_cast<float>(c.redF()), static_cast<float>(c.greenF()),
                                   static_cast<float>(c.blueF())});
}

void NetPlayDialog::AppendChat(const std::string& msg)
{
  DisplayMessage(QString::fromStdString(msg), "");
  QApplication::alert(this);
}

void NetPlayDialog::DisplayActiveGeckoCodes()
{
  if (!IsHosting())
  {
    return;
  }
  Settings::Instance().GetNetPlayClient()->GetActiveGeckoCodes();
} 

void NetPlayDialog::OnMsgChangeGame(const NetPlay::SyncIdentifier& sync_identifier,
                                    const std::string& netplay_name)
{
  QString qname = QString::fromStdString(netplay_name);
  QueueOnObject(this, [this, qname, netplay_name, &sync_identifier] {
    m_game_button->setText(qname);
    m_current_game_identifier = sync_identifier;
    m_current_game_name = netplay_name;
    UpdateDiscordPresence();
  });
  DisplayMessage(tr("Game changed to \"%1\"").arg(qname), "magenta");
}

void NetPlayDialog::OnMsgChangeGBARom(int pad, const NetPlay::GBAConfig& config)
{
  if (config.has_rom)
  {
    DisplayMessage(
        tr("GBA%1 ROM changed to \"%2\"").arg(pad + 1).arg(QString::fromStdString(config.title)),
        "magenta");
  }
  else
  {
    DisplayMessage(tr("GBA%1 ROM disabled").arg(pad + 1), "magenta");
  }
}

void NetPlayDialog::GameStatusChanged(bool running)
{
  QueueOnObject(this, [this, running] { SetOptionsEnabled(!running); });
}

void NetPlayDialog::SetOptionsEnabled(bool enabled)
{
  if (Settings::Instance().GetNetPlayServer())
  {
    m_start_button->setEnabled(enabled);
    m_game_button->setEnabled(enabled);
    m_load_wii_action->setEnabled(enabled);
    m_write_save_data_action->setEnabled(enabled);
    m_sync_save_data_action->setEnabled(enabled);
    // m_sync_codes_action->setEnabled(enabled);
    m_assign_ports_button->setEnabled(enabled);
    m_strict_settings_sync_action->setEnabled(enabled);
    m_sync_all_wii_saves_action->setEnabled(enabled && m_sync_save_data_action->isChecked());

    // Only show fair input delay on the network menu
    m_fixed_delay_action->setEnabled(enabled);
    // m_host_input_authority_action->setEnabled(enabled);
    // m_golf_mode_action->setEnabled(enabled);
    m_ranked_box->setEnabled(enabled);
  }

  m_record_input_action->setEnabled(enabled);
}

void NetPlayDialog::RankedStartingMsg(bool is_ranked)
{
  StateAuxillary::setIsRanked(is_ranked);
  if (is_ranked)
  {
    DisplayMessage(tr("NOTE: Ranked is Enabled. Community standard gecko codes are enabled and all others are disabled. 10 "),
                   "mediumseagreen");
  }
  else
  {
    DisplayMessage(
        tr("NOTE: Ranked Mode is Disabled. Custom gecko codes may be enabled."),
        "crimson");
  }
}

void NetPlayDialog::OnMsgStartGame()
{
  DisplayMessage(tr("Started game"), "green");

  g_netplay_chat_ui =
      std::make_unique<NetPlayChatUI>([this](const std::string& message) { SendMessage(message); });

  if (m_host_input_authority &&
      Settings::Instance().GetNetPlayClient()->GetNetSettings().m_GolfMode)
  {
    g_netplay_golf_ui = std::make_unique<NetPlayGolfUI>(Settings::Instance().GetNetPlayClient());
  }

  QueueOnObject(this, [this] {
    auto client = Settings::Instance().GetNetPlayClient();

    if (client)
    {
      if (auto game = FindGameFile(m_current_game_identifier))
        client->StartGame(game->GetFilePath());
      else
        PanicAlertFmtT("Selected game doesn't exist in game list!");
    }
    UpdateDiscordPresence();
  });
}

void NetPlayDialog::OnMsgStopGame()
{
  g_netplay_chat_ui.reset();
  g_netplay_golf_ui.reset();
  QueueOnObject(this, [this] { UpdateDiscordPresence(); });
}

void NetPlayDialog::OnMsgPowerButton()
{
  if (!Core::IsRunning())
    return;
  QueueOnObject(this, [] { UICommon::TriggerSTMPowerEvent(); });
}

void NetPlayDialog::OnPlayerConnect(const std::string& player)
{
  DisplayMessage(tr("%1 has joined").arg(QString::fromStdString(player)), "darkcyan");
  // If host, Call SendRankedState to send the ranked message to everyone
  // This will probably introduce a weird UI effect where people who are up to date with the value still receive the UI update that ranked is disabled/enabled
  if (IsHosting())
  {
    INFO_LOG_FMT(NETPLAY, "Non-host has joined -- sending the current ranked value of {} to them", m_current_ranked_value);
    Settings::Instance().GetNetPlayClient()->SendRankedState(m_current_ranked_value);
  }
}

void NetPlayDialog::OnPlayerDisconnect(const std::string& player)
{
  DisplayMessage(tr("%1 has left").arg(QString::fromStdString(player)), "darkcyan");
}

void NetPlayDialog::OnPadBufferChanged(u32 buffer)
{
  QueueOnObject(this, [this, buffer] {
    const QSignalBlocker blocker(m_buffer_size_box);
    m_buffer_size_box->setValue(buffer);
  });
  DisplayMessage(m_host_input_authority ? tr("Max buffer size changed to %1").arg(buffer) :
                                          tr("Buffer size changed to %1").arg(buffer),
                 "darkcyan");

  m_buffer_size = static_cast<int>(buffer);
}

void NetPlayDialog::OnHostInputAuthorityChanged(bool enabled)
{
  m_host_input_authority = enabled;
  DisplayMessage(enabled ? tr("Host input authority enabled") : tr("Host input authority disabled"),
                 "");

  QueueOnObject(this, [this, enabled] {
    const bool is_hosting = IsHosting();
    const bool enable_buffer = is_hosting != enabled;

    if (is_hosting)
    {
      m_buffer_size_box->setEnabled(enable_buffer);
      m_buffer_label->setEnabled(enable_buffer);
      m_buffer_size_box->setHidden(false);
      m_buffer_label->setHidden(false);
    }
    else
    {
      m_buffer_size_box->setEnabled(true);
      m_buffer_label->setEnabled(true);
      m_buffer_size_box->setHidden(!enable_buffer);
      m_buffer_label->setHidden(!enable_buffer);
    }

    m_buffer_label->setText(enabled ? tr("Max Buffer:") : tr("Buffer:"));
    if (enabled)
    {
      const QSignalBlocker blocker(m_buffer_size_box);
      m_buffer_size_box->setValue(Config::Get(Config::NETPLAY_CLIENT_BUFFER_SIZE));
    }
  });
}

void NetPlayDialog::OnDesync(u32 frame, const std::string& player)
{
  DisplayMessage(tr("Possible desync detected: %1 might have desynced at frame %2")
                     .arg(QString::fromStdString(player), QString::number(frame)),
                 "red", OSD::Duration::VERY_LONG);
  OSD::AddTypedMessage(OSD::MessageType::NetPlayDesync,
                       "Possible desync detected. Game restart advised.", OSD::Duration::VERY_LONG,
                       OSD::Color::RED);
}

void NetPlayDialog::OnConnectionLost()
{
  DisplayMessage(tr("Lost connection to NetPlay server..."), "red");
}

void NetPlayDialog::OnConnectionError(const std::string& message)
{
  QueueOnObject(this, [this, message] {
    ModalMessageBox::critical(this, tr("Error"),
                              tr("Failed to connect to server: %1").arg(tr(message.c_str())));
  });
}

void NetPlayDialog::OnTraversalError(TraversalClient::FailureReason error)
{
  QueueOnObject(this, [this, error] {
    switch (error)
    {
    case TraversalClient::FailureReason::BadHost:
      ModalMessageBox::critical(this, tr("Traversal Error"), tr("Couldn't look up central server"));
      QDialog::reject();
      break;
    case TraversalClient::FailureReason::VersionTooOld:
      ModalMessageBox::critical(this, tr("Traversal Error"),
                                tr("Dolphin is too old for traversal server"));
      QDialog::reject();
      break;
    case TraversalClient::FailureReason::ServerForgotAboutUs:
    case TraversalClient::FailureReason::SocketSendError:
    case TraversalClient::FailureReason::ResendTimeout:
      UpdateGUI();
      break;
    }
  });
}

void NetPlayDialog::OnTraversalStateChanged(TraversalClient::State state)
{
  switch (state)
  {
  case TraversalClient::State::Connected:
  case TraversalClient::State::Failure:
    UpdateDiscordPresence();
    break;
  default:
    break;
  }
}

void NetPlayDialog::OnGameStartAborted()
{
  QueueOnObject(this, [this] { SetOptionsEnabled(true); });
}

void NetPlayDialog::OnGolferChanged(const bool is_golfer, const std::string& golfer_name)
{
  if (m_host_input_authority)
  {
    QueueOnObject(this, [this, is_golfer] {
      m_buffer_size_box->setEnabled(!is_golfer);
      m_buffer_label->setEnabled(!is_golfer);
    });
  }

  if (!golfer_name.empty())
    DisplayMessage(tr("%1 is now golfing").arg(QString::fromStdString(golfer_name)), "");
}

void NetPlayDialog::OnLoginError(CitrusRequest::LoginError error)
{
  std::string userJSONPath = File::GetCitrusUserFilePath();
  std::string citrusLauncherEXEPath = File::GetCitrusLauncherEXEPath();
  switch (error)
  {
  case CitrusRequest::LoginError::ServerError:
  {
    ModalMessageBox::critical(
        this, tr("Error"),
        tr("Failed to log in: %1").arg(tr(CitrusRequest::loginErrorMap.at(error).c_str())));
    QDialog::reject();
    break;
  }
  case CitrusRequest::LoginError::InvalidLogin:
  case CitrusRequest::LoginError::InvalidUserId:
  {
    // Great naming conventions coming up
    QDialog* dialog1 = new QDialog(this);
    dialog1->setWindowTitle(tr("Error"));
    auto* label1 = new QLabel(
        tr("Failed to log in: %1").arg(tr(CitrusRequest::loginErrorMap.at(error).c_str())));
    label1->setTextFormat(Qt::RichText);

    auto* buttons1 = new QDialogButtonBox;
    auto* projectcitrus1 = buttons1->addButton(tr("Open User File"), QDialogButtonBox::AcceptRole);

    connect(projectcitrus1, &QPushButton::clicked, this, [userJSONPath, dialog1]() {
      std::string pathToAppData = "\"" + userJSONPath + "\"";
      ShellExecuteA(NULL, "open", &pathToAppData[0], NULL, NULL, SW_HIDE);
      dialog1->close();
    });

    auto* layout1 = new QVBoxLayout;
    layout1->addWidget(label1);
    dialog1->setLayout(layout1);
    layout1->addWidget(buttons1);

    dialog1->exec();
    break;
  }
  case CitrusRequest::LoginError::NoUserFile:
    QDialog* dialog = new QDialog(this);
    dialog->setWindowTitle(tr("Error"));
    auto* label =
        new QLabel(tr("<h2>No user file found.</h2><h4>Launch Citrus Launcher "
                      "to log in.</h4>"
                      "<p>Upon opening Citrus Launcher, log in can be completed from the top-right "
                      "of the navigation bar.</p>"
                      "<em>Developer info: The user file was not found at either: %1 </em>")
                       .arg(tr(File::ListPossibleCitrusUserFilePaths().c_str())));
    label->setTextInteractionFlags(Qt::TextSelectableByMouse);
    label->setTextFormat(Qt::RichText);

    auto* buttons = new QDialogButtonBox;
    auto* projectcitrus =
        buttons->addButton(tr("Launch Citrus Launcher"), QDialogButtonBox::AcceptRole);

    connect(projectcitrus, &QPushButton::clicked, this, [citrusLauncherEXEPath]() {
      std::string pathToAppData = "\"" + citrusLauncherEXEPath + "\"";
      STARTUPINFO si;
      PROCESS_INFORMATION pi;
      memset(&si, 0, sizeof(si));
      si.cb = sizeof(si);
      CreateProcessA(NULL, &pathToAppData[0], NULL, NULL, TRUE, CREATE_NO_WINDOW, NULL, NULL,
                     (LPSTARTUPINFOA)&si, &pi);
    });

    auto* layout = new QVBoxLayout;
    layout->addWidget(label);
    dialog->setLayout(layout);
    layout->addWidget(buttons);

    dialog->exec();
    break;
  }

}

bool NetPlayDialog::IsRecording()
{
  std::optional<bool> is_recording = RunOnObject(m_record_input_action, &QAction::isChecked);
  if (is_recording)
    return *is_recording;
  return false;
}

std::shared_ptr<const UICommon::GameFile>
NetPlayDialog::FindGameFile(const NetPlay::SyncIdentifier& sync_identifier,
                            NetPlay::SyncIdentifierComparison* found)
{
  NetPlay::SyncIdentifierComparison temp;
  if (!found)
    found = &temp;

  *found = NetPlay::SyncIdentifierComparison::DifferentGame;

  std::optional<std::shared_ptr<const UICommon::GameFile>> game_file =
      RunOnObject(this, [this, &sync_identifier, found] {
        for (int i = 0; i < m_game_list_model.rowCount(QModelIndex()); i++)
        {
          auto file = m_game_list_model.GetGameFile(i);
          *found = std::min(*found, file->CompareSyncIdentifier(sync_identifier));
          if (*found == NetPlay::SyncIdentifierComparison::SameGame)
            return file;
        }
        return static_cast<std::shared_ptr<const UICommon::GameFile>>(nullptr);
      });
  if (game_file)
    return *game_file;
  return nullptr;
}

std::string NetPlayDialog::FindGBARomPath(const std::array<u8, 20>& hash, std::string_view title,
                                          int device_number)
{
#ifdef HAS_LIBMGBA
  auto result = RunOnObject(this, [&, this] {
    std::string rom_path;
    std::array<u8, 20> rom_hash;
    std::string rom_title;
    for (size_t i = device_number; i < static_cast<size_t>(device_number) + 4; ++i)
    {
      rom_path = Config::Get(Config::MAIN_GBA_ROM_PATHS[i % 4]);
      if (!rom_path.empty() && HW::GBA::Core::GetRomInfo(rom_path.c_str(), rom_hash, rom_title) &&
          rom_hash == hash && rom_title == title)
      {
        return rom_path;
      }
    }
    while (!(rom_path = GameCubePane::GetOpenGBARom(title)).empty())
    {
      if (HW::GBA::Core::GetRomInfo(rom_path.c_str(), rom_hash, rom_title))
      {
        if (rom_hash == hash && rom_title == title)
          return rom_path;
        ModalMessageBox::critical(
            this, tr("Error"),
            QString::fromStdString(Common::FmtFormatT(
                "Mismatched ROMs\n"
                "Selected: {0}\n- Title: {1}\n- Hash: {2:02X}\n"
                "Expected:\n- Title: {3}\n- Hash: {4:02X}",
                rom_path, rom_title, fmt::join(rom_hash, ""), title, fmt::join(hash, ""))));
      }
      else
      {
        ModalMessageBox::critical(
            this, tr("Error"), tr("%1 is not a valid ROM").arg(QString::fromStdString(rom_path)));
      }
    }
    return std::string();
  });
  if (result)
    return *result;
#endif
  return {};
}

void NetPlayDialog::LoadSettings()
{
  const int buffer_size = Config::Get(Config::NETPLAY_BUFFER_SIZE);
  const bool write_save_data = Config::Get(Config::NETPLAY_WRITE_SAVE_DATA);
  const bool load_wii_save = Config::Get(Config::NETPLAY_LOAD_WII_SAVE);
  const bool sync_saves = Config::Get(Config::NETPLAY_SYNC_SAVES);
  // const bool sync_codes = Config::Get(Config::NETPLAY_SYNC_CODES);
  const bool record_inputs = Config::Get(Config::NETPLAY_RECORD_INPUTS);
  const bool strict_settings_sync = Config::Get(Config::NETPLAY_STRICT_SETTINGS_SYNC);
  const bool sync_all_wii_saves = Config::Get(Config::NETPLAY_SYNC_ALL_WII_SAVES);
  const bool golf_mode_overlay = Config::Get(Config::NETPLAY_GOLF_MODE_OVERLAY);
  const bool hide_remote_gbas = Config::Get(Config::NETPLAY_HIDE_REMOTE_GBAS);

  m_buffer_size_box->setValue(buffer_size);
  m_write_save_data_action->setChecked(write_save_data);
  m_load_wii_action->setChecked(load_wii_save);
  m_sync_save_data_action->setChecked(sync_saves);
  // m_sync_codes_action->setChecked(sync_codes);
  m_record_input_action->setChecked(record_inputs);
  m_strict_settings_sync_action->setChecked(strict_settings_sync);
  m_sync_all_wii_saves_action->setChecked(sync_all_wii_saves);
  m_golf_mode_overlay_action->setChecked(golf_mode_overlay);
  m_hide_remote_gbas_action->setChecked(hide_remote_gbas);

  // const std::string network_mode = Config::Get(Config::NETPLAY_NETWORK_MODE);

  // if (network_mode == "fixeddelay")
  // {
  //   m_fixed_delay_action->setChecked(true);
  // }
  // else if (network_mode == "hostinputauthority")
  // {
  //   m_host_input_authority_action->setChecked(true);
  // }
  // else if (network_mode == "golf")
  // {
  //   m_golf_mode_action->setChecked(true);
  // }
  // else
  // {
  //   WARN_LOG_FMT(NETPLAY, "Unknown network mode '{}', using 'fixeddelay'", network_mode);
  //   m_fixed_delay_action->setChecked(true);
  // }

  m_fixed_delay_action->setChecked(true);
}

void NetPlayDialog::SaveSettings()
{
  Config::ConfigChangeCallbackGuard config_guard;

  if (m_host_input_authority)
    Config::SetBase(Config::NETPLAY_CLIENT_BUFFER_SIZE, m_buffer_size_box->value());
  else
    Config::SetBase(Config::NETPLAY_BUFFER_SIZE, m_buffer_size_box->value());

  Config::SetBase(Config::NETPLAY_WRITE_SAVE_DATA, m_write_save_data_action->isChecked());
  Config::SetBase(Config::NETPLAY_LOAD_WII_SAVE, m_load_wii_action->isChecked());
  Config::SetBase(Config::NETPLAY_SYNC_SAVES, m_sync_save_data_action->isChecked());
  // Config::SetBase(Config::NETPLAY_SYNC_CODES, m_sync_codes_action->isChecked());
  Config::SetBase(Config::NETPLAY_RECORD_INPUTS, m_record_input_action->isChecked());
  Config::SetBase(Config::NETPLAY_STRICT_SETTINGS_SYNC, m_strict_settings_sync_action->isChecked());
  Config::SetBase(Config::NETPLAY_SYNC_ALL_WII_SAVES, m_sync_all_wii_saves_action->isChecked());
  Config::SetBase(Config::NETPLAY_GOLF_MODE_OVERLAY, m_golf_mode_overlay_action->isChecked());
  Config::SetBase(Config::NETPLAY_HIDE_REMOTE_GBAS, m_hide_remote_gbas_action->isChecked());

  // std::string network_mode;
  // if (m_fixed_delay_action->isChecked())
  // {
  //   network_mode = "fixeddelay";
  // }
  // else if (m_host_input_authority_action->isChecked())
  // {
  //   network_mode = "hostinputauthority";
  // }
  // else if (m_golf_mode_action->isChecked())
  // {
  //   network_mode = "golf";
  // }

  // Config::SetBase(Config::NETPLAY_NETWORK_MODE, network_mode);

  Config::SetBase(Config::NETPLAY_NETWORK_MODE, "fixeddelay");
}

void NetPlayDialog::ShowMD5Dialog(const std::string& title)
{
  QueueOnObject(this, [this, title] {
    m_md5_menu->setEnabled(false);

    if (m_md5_dialog->isVisible())
      m_md5_dialog->close();

    m_md5_dialog->show(QString::fromStdString(title));
  });
}

void NetPlayDialog::SetMD5Progress(int pid, int progress)
{
  QueueOnObject(this, [this, pid, progress] {
    if (m_md5_dialog->isVisible())
      m_md5_dialog->SetProgress(pid, progress);
  });
}

void NetPlayDialog::SetMD5Result(int pid, const std::string& result)
{
  QueueOnObject(this, [this, pid, result] {
    m_md5_dialog->SetResult(pid, result);
    m_md5_menu->setEnabled(true);
  });
}

void NetPlayDialog::AbortMD5()
{
  QueueOnObject(this, [this] {
    m_md5_dialog->close();
    m_md5_menu->setEnabled(true);
  });
}

void NetPlayDialog::ShowChunkedProgressDialog(const std::string& title, const u64 data_size,
                                              const std::vector<int>& players)
{
  QueueOnObject(this, [this, title, data_size, players] {
    if (m_chunked_progress_dialog->isVisible())
      m_chunked_progress_dialog->done(QDialog::Accepted);

    m_chunked_progress_dialog->show(QString::fromStdString(title), data_size, players);
  });
}

void NetPlayDialog::HideChunkedProgressDialog()
{
  QueueOnObject(this, [this] { m_chunked_progress_dialog->done(QDialog::Accepted); });
}

void NetPlayDialog::SetChunkedProgress(const int pid, const u64 progress)
{
  QueueOnObject(this, [this, pid, progress] {
    if (m_chunked_progress_dialog->isVisible())
      m_chunked_progress_dialog->SetProgress(pid, progress);
  });
}

void NetPlayDialog::SetHostWiiSyncData(std::vector<u64> titles, std::string redirect_folder)
{
  auto client = Settings::Instance().GetNetPlayClient();
  if (client)
    client->SetWiiSyncData(nullptr, std::move(titles), std::move(redirect_folder));
}
